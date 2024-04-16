import torch, os
import pathlib as pb
import nibabel as nib
import numpy as np
import torchio as tio

from tqdm import tqdm
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LinearLR
from monai.networks.nets.dynunet import DynUNet
from torch.utils.data import Dataset, DataLoader
from medpy.metric.binary import dc, precision, recall
from tensorboardX import SummaryWriter


def conserve_only_certain_labels(label, designated_labels=[105, 106, 107, 108, 109]):
    # 6: stomach, 57: colon
    if designated_labels is None:
        return label.long()
    label_ = np.zeros_like(label, dtype=np.uint8)
    for il, l in enumerate(designated_labels):
        label_[label == l] = il + 1
    return label_


class ColonSegmentation(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        self.files = "/mnt/data/oss_beijing/dailinrui/data/cmu/splitall/mask"
        self.ds = list(pb.Path(self.files).glob("*.nii.gz"))
        self.train_ds = self.ds[:round(.8 * len(self.ds))]
        self.val_ds = self.ds[round(.8 * len(self.ds)):]
        self.split_ds = self.train_ds if split == "train" else self.val_ds
        
        self.resize = tio.transforms.Resize((16, 64, 64))
        self.transforms = tio.transforms.RandomAffine(scales=(0.9, 1.1), degrees=0, translation=(0, 10))
    
    @staticmethod
    def load_fn(x):
        return nib.load(x).dataobj[:].transpose(2, 1, 0)
        
    def __len__(self):
        return len(self.split_ds)
    
    def __getitem__(self, index):
        sample = self.split_ds[index]
        sample = self.load_fn(sample)
        
        s, e = np.where(sample.sum((1, 2)))[0][[0, -1]]
        segmented = conserve_only_certain_labels(sample[s: e + 1])
        unsegmented = (segmented > 0).astype(np.uint8)
        
        subject = tio.Subject(seg=tio.LabelMap(tensor=segmented[None]), raw=tio.LabelMap(tensor=unsegmented[None]))
        subject = self.resize(subject)
        subject = self.transforms(subject)
        
        return {"seg": subject.seg.data, "raw": subject.raw.data}
        
        
class Trainer:
    legends = ["background", "ascendant", "transversal", "descendant", "sigmoid", "rectum"]
    def __init__(self, cuda_index=0) -> None:
        self.lr = 1e-3
        self.batch_size = 24
        self.max_epochs = 100
        self.best = 0
        self.cuda_index = cuda_index
        
        self._model_path = "/mnt/workspace/dailinrui/data/pretrained/ccdm/segmentation/colon_segment/model"
        self._tensorboard_path = "/mnt/workspace/dailinrui/data/pretrained/ccdm/segmentation/colon_segment/logs"
        
        self.train_ds = ColonSegmentation()
        self.val_ds = ColonSegmentation("val")
        
        self.train_it = tqdm(range(self.max_epochs), desc="train progress")
        self.train_wr = SummaryWriter(self._tensorboard_path)
        self.val_proc = None
        self.val_dict = torch.multiprocessing.Manager().dict()
        
        self.train_dl = DataLoader(self.train_ds, self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        self.val_dl = DataLoader(self.val_ds, batch_size=1, pin_memory=True, num_workers=1)
        
        self.model = DynUNet(spatial_dims=3,
                             in_channels=1,
                             out_channels=6,
                             channels=(16, 32, 64, 128, 256),
                             strides=(2, 2, 2, 2),).cuda(self.cuda_index)
        self.model.train()
        
        self.optim = SGD(self.model.parameters(), self.lr, momentum=.99)
        self.sched = LinearLR(self.optim, start_factor=1, total_iters=self.max_epochs)
        self.loss = torch.nn.CrossEntropyLoss()
        
    def train(self):
        for ep in self.train_it:
            self._train(ep)
            self.sched.step()
            if ep % 2 == 0:
                if self.val_proc is not None:
                    self.val_proc.join()
                    self.best = self.val_dict.get("best", self.best)
                    
                self.val_proc = torch.multiprocessing.get_context("spawn").Process(target=self.val,
                                                                                   args=(self.model.eval(), self.optim, ep, self.val_dict, self.best, self.val_dl, self.cuda_index, self._tensorboard_path, self._model_path, self._save))
                self.val_proc.start()
        self._save(self._model_path, self.best, self.model, self.optim, "last.ckpt")
            
    def _train(self, ep):
        for itr, batch in enumerate(self.train_dl):
            seg, raw = batch["seg"].cuda(self.cuda_index), batch["raw"].cuda(self.cuda_index)
            
            out = self.model(raw)
            soft_out = seg.softmax(1)
            pred = soft_out.argmax(1)
            loss = self.loss(out, seg[:, 0].long())
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            self.train_wr.add_scalars("train/dice",
                                      {self.legends[k]: dc(pred.cpu().numpy() == k, seg.cpu().numpy() == k) for k in range(1, 6)},
                                      itr + len(self.train_dl) * ep)
            self.train_wr.add_scalars("train/precision",
                                      {self.legends[k]: precision(pred.cpu().numpy() == k, seg.cpu().numpy() == k) for k in range(1, 6)},
                                      itr + len(self.train_dl) * ep)
            self.train_it.set_postfix(itr=itr + len(self.train_dl) * ep, loss=loss.item())
    
    @staticmethod
    def val(model, optim, ep, val_dict, best, val_dl, cuda_index, _tensorboard_path, _model_path, _save):
        
        writer = SummaryWriter(_tensorboard_path)
        met = np.zeros((len(val_dl), 5, 3))
        val_it = tqdm(enumerate(val_dl), total=len(val_dl), desc='validation progress')
        
        for itr, batch in val_it:
            seg, raw = batch["seg"].cuda(cuda_index), batch["raw"].cuda(cuda_index)
            pred = model(raw).softmax(1).argmax(1)
            met[itr] = np.array([[fn(seg.cpu().numpy() == i, pred.cpu().numpy() == i) for fn in [dc, precision, recall]] for i in range(1, 6)])
            val_it.set_postfix(dice=met[..., 0].sum() / (itr + 1), precision=met[..., 1].sum() / (itr + 1), recall=met[..., 2].sum() / (itr + 1))
            
        for index, name in enumerate(['dice', 'precision', 'recall']):
            writer.add_scalars(f'val/{name}', {f'{Trainer.legends[i]}': met[:, i-1, index].mean() for i in range(1, 6)}, ep)
            writer.add_scalars(f'val/{name}', {f'average': met[:, -1, index].mean()}, ep)
            
        if (new_best := met[..., 0].mean()) > best:
            print(f"best for epoch {ep}: {new_best:.2f}")
            _save(_model_path, new_best, model, optim, f"best_model_dice={round(new_best, 2)}.ckpt")
            val_dict["best"] = new_best
            
        writer.close()
    
    @staticmethod
    def _save(_save_path, best=0, model=None, optim=None, _save_name=None):
        torch.save(dict(model=model.state_dict(),
                        optimizer=optim.state_dict(),
                        current_best_dice=best), os.path.join(_save_path, _save_name))
        
    def _load(self, _load_path):
        obj = torch.load(_load_path, map_location="cpu")
        self.model.load_state_dict(obj["model"])
        self.optim.load_state_dict(obj["optimizer"])
        self.best = obj["current_best_dice"]
    
    def test(self, input_folder_or_file, output_folder_or_file):
        self.model.eval()
        print(f"best dice on validation {self.best}")
        if os.path.isfile(input_folder_or_file):
            input_folder_or_file = [input_folder_or_file]
        else:
            input_folder_or_file = list(pb.Path(input_folder_or_file).glob("*.nii.gz"))
        for case in input_folder_or_file:
            raw = torch.tensor(ColonSegmentation.load_fn(case)).cuda(self.cuda_index)
            seg = self._test(raw)
            nib.save(nib.nifti1.Nifti1Image(dataobj=seg.transpose(2, 1, 0), affine=None),
                     str(case).replace(input_folder_or_file, output_folder_or_file))
            
    def _test(self, tensor):
        tensor = tensor[None, None]
        return self.model(tensor).softmax(1).argmax(1).cpu().numpy()[0]
        
    def loss(self, colon_mask, tumor_mask, gt):
        with torch.no_grad():
            seg = self._test(colon_mask)
        cgt = seg == gt + 1
        dist = .01 * (torch.argwhere(cgt).mean(0) - torch.argwhere(tumor_mask).mean(0)) ** 2
        return dist


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
            