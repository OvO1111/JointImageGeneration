# Joint CT & Anatomical Mask Generation
Official implementation of GuideGen in `GuideGen: A Text-guided Framework for Joint CT Volume and Anatomical structure Generation`, submitted to MICCAI 2024.

## Code
Relevant training & test code will be made publicly available shortly <3. You can familiarize yourself with the methodology of [CCDM](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation) and [LDM](https://github.com/CompVis/latent-diffusion) first. These are modified and combined in our paper to cope with our goal of joint CT & anatomical mask generation.

## Teasers
<figure>
<figcaption align = "center"><b>Overall pipeline for GuideGen</b>. At inference time, given a text condition (white), the volumetric mask sampler (yellow) outputs a corresponding segmentation mask for major abdominal organs and tumor site. This mask is upsampled and sliced before passing into the conditional image generator (blue) to generate the CT volume autoregressively.</figcaption>
<img src="https://github.com/OvO1111/JointImageGeneration/assets/43473365/9a310e25-d7d8-4613-b8f8-4f1e2df0cf3c" alt="pipeline" style="width:100%">
</figure>

<figure>
<figcaption align = "center"><b>Comparative Results with other Methods</b>. Colored regions on the right represent different organ masks and green area marked in each figure represents generated tumor site. Green, red and blue boxes on generated masks represent synthesized tumor masks that is well-positioned, misplaced or missing with respect to the text condition (which is exhibited as real tumor locations in the Original Anatomies). </figcaption>
<img src="https://github.com/OvO1111/JointImageGeneration/assets/43473365/949ca811-1b26-4676-88e1-b6942e032cfe" alt="qualitative_results" style="width:100%">
</figure>


