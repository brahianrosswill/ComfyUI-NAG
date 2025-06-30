# Node Documentation

This document provides a detailed description of the custom nodes available in this repository, their parameters, and their practical effects on image generation.

## `NAGCFGGuider`

The `NAGCFGGuider` node is responsible for setting up the Negative Guidance (NAG) parameters along with the standard Classifier-Free Guidance (CFG) for the diffusion model. It prepares a "guider" object that influences how the model interprets prompts and generates images.

**Category:** `sampling/custom_sampling/guiders`

**Function:** `get_guider`

### Input Parameters:

1.  **`model`**:
    *   **Type:** `MODEL`
    *   **Description:** The diffusion model to be guided. This is the base model that will be used for the generation process.
    *   **Practical Effect:** Determines the fundamental characteristics and style of the generated image.

2.  **`positive`**:
    *   **Type:** `CONDITIONING`
    *   **Description:** The positive conditioning (prompt) that describes the desired attributes of the image.
    *   **Practical Effect:** Guides the model towards generating an image that includes elements and concepts specified in this prompt.

3.  **`negative`**:
    *   **Type:** `CONDITIONING`
    *   **Description:** The standard negative conditioning (prompt) that describes attributes to be excluded from the image.
    *   **Practical Effect:** Steers the model away from generating an image with elements and concepts specified in this prompt. This is the traditional negative prompt used in CFG.

4.  **`nag_negative`**:
    *   **Type:** `CONDITIONING`
    *   **Description:** The specific negative conditioning for the Negative Guidance technique. This prompt is used by NAG to further refine the image by pushing it away from these undesired concepts, often with a different emphasis or mechanism than the standard negative prompt.
    *   **Practical Effect:** Allows for more nuanced control over undesired attributes. For example, while the standard `negative` prompt might broadly exclude "blurriness," the `nag_negative` prompt could target a specific type of artifact or a more subtle unwanted feature.

5.  **`cfg`**:
    *   **Type:** `FLOAT`
    *   **Default:** `1.0`
    *   **Min:** `0.0`
    *   **Max:** `100.0`
    *   **Step:** `0.1`
    *   **Round:** `0.01`
    *   **Description:** Classifier-Free Guidance scale. This parameter controls how strongly the model should adhere to the `positive` prompt.
    *   **Practical Effect:** Higher values make the image more closely match the `positive` prompt, potentially at the cost of coherence or image quality if set too high. Lower values allow for more creativity and deviation from the prompt. When used with NAG, this `cfg` value works in concert with the NAG parameters.

6.  **`nag_scale`**:
    *   **Type:** `FLOAT`
    *   **Default:** `5.0`
    *   **Min:** `0.0`
    *   **Max:** `100.0`
    *   **Step:** `0.1`
    *   **Round:** `0.01`
    *   **Description:** The main scaling factor for Negative Guidance. It controls the overall strength of the `nag_negative` prompt's influence.
    *   **Practical Effect:** Higher values will more aggressively push the image away from the concepts in `nag_negative`. This can be useful for strongly suppressing unwanted features or styles. Too high a value might introduce artifacts or overly distort the image.

7.  **`nag_tau`**:
    *   **Type:** `FLOAT`
    *   **Default:** `2.5`
    *   **Min:** `1.0`
    *   **Max:** `10.0`
    *   **Step:** `0.1`
    *   **Round:** `0.01`
    *   **Description:** Tau parameter for Negative Guidance. This often relates to a threshold or a sensitivity factor in how NAG is applied, potentially influencing how much deviation from the `nag_negative` prompt is tolerated before correction kicks in. The exact mathematical interpretation can vary based on the specific NAG implementation in the underlying samplers and models.
    *   **Practical Effect:** Adjusting `nag_tau` can fine-tune the behavior of NAG. A higher tau might make the NAG effect more pronounced or apply it more broadly across the diffusion timesteps. It can influence the "distance" the generation is pushed from the `nag_negative` conditioning. For example, a higher `nag_tau` might make the negative guidance more assertive in pushing away from the `nag_negative` prompt, even for subtle similarities.

8.  **`nag_alpha`**:
    *   **Type:** `FLOAT`
    *   **Default:** `0.25`
    *   **Min:** `0.0`
    *   **Max:** `1.0`
    *   **Step:** `0.01`
    *   **Round:** `0.01`
    *   **Description:** Alpha parameter for Negative Guidance. This parameter often acts as a blending factor or a dampening factor for the NAG correction. It can control the intensity of the NAG adjustment at each step or across certain stages of the diffusion process.
    *   **Practical Effect:** `nag_alpha` allows for finer control over the NAG intensity. A lower alpha might apply a gentler correction, while a higher alpha could make the correction stronger. It can be used to prevent over-correction or to modulate the NAG effect, ensuring it doesn't overpower the positive guidance or introduce unwanted artifacts. For instance, it might control how much the "corrected" latent (pushed away from `nag_negative`) is mixed with the original latent.

9.  **`nag_sigma_end`**:
    *   **Type:** `FLOAT`
    *   **Default:** `0.0`
    *   **Min:** `0.0`
    *   **Max:** `20.0`
    *   **Step:** `0.01`
    *   **Round:** `0.01`
    *   **Description:** Defines the sigma (noise level) at which the Negative Guidance effect stops or is no longer applied. In diffusion models, sigma decreases as the image is denoised.
    *   **Practical Effect:** This parameter controls how long into the diffusion process NAG remains active. A value of `0.0` (default) might imply NAG is active throughout most or all of the process. Setting a higher `nag_sigma_end` means NAG will stop influencing the generation earlier (when noise levels are still higher). This can be useful to apply NAG for correcting major compositional elements or broad undesired features early on, without affecting finer details that emerge in later stages of denoising. For example, if `nag_negative` is "extra limbs", applying NAG only at higher sigma values (earlier in the process) might be sufficient.

10. **`latent_image`**:
    *   **Type:** `LATENT`
    *   **Description:** The input latent image. While the `NAGCFGGuider` primarily sets up guidance parameters, it requires the latent image to determine the batch size for the guider.
    *   **Practical Effect:** Ensures the guider is configured correctly for the number of images being processed in the current batch. The content of the latent itself is primarily used by the sampler, not directly manipulated by this guider node beyond batch size determination.

### Output:

*   **`GUIDER`**: An object containing the configured guidance settings (CFG and NAG) to be used by a custom sampler like `KSamplerWithNAG` or `SamplerCustomAdvanced`.

### How NAG Parameters Interact:

The `nag_` parameters work together to control the Negative Guidance mechanism:
*   `nag_negative`: Defines *what* to avoid.
*   `nag_scale`: Defines *how strongly* to avoid it.
*   `nag_tau` and `nag_alpha`: Fine-tune the *behavior and intensity* of the avoidance. `nag_tau` can be seen as a threshold for how "far" the generation needs to be from the negative concept, while `nag_alpha` can control the strength of the corrective "push" once that threshold is met or how the correction is blended.
*   `nag_sigma_end`: Defines *when* to stop avoiding it during the denoising process.

Experimenting with these parameters allows for sophisticated control over the final image, helping to eliminate unwanted artifacts, styles, or objects that might be difficult to remove with standard negative prompts alone.
---

## `KSamplerWithNAG`

The `KSamplerWithNAG` node is a custom KSampler that integrates Negative Guidance (NAG) capabilities directly into the sampling process. It takes standard sampling parameters along with NAG-specific parameters to generate a latent image. This node combines the roles of a traditional sampler and a NAG guider, offering a streamlined way to use NAG.

**Category:** `sampling`

**Function:** `sample`

**Description:** "Uses the provided model, positive and negative conditioning to denoise the latent image." (Incorporates NAG)

### Input Parameters:

1.  **`model`**:
    *   **Type:** `MODEL`
    *   **Tooltip:** "The model used for denoising the input latent."
    *   **Description:** The diffusion model that will generate the image.
    *   **Practical Effect:** Same as in `NAGCFGGuider`; determines the base characteristics of the output.

2.  **`seed`**:
    *   **Type:** `INT`
    *   **Default:** `0`
    *   **Min:** `0`
    *   **Max:** `0xffffffffffffffff`
    *   **`control_after_generate`:** `True`
    *   **Tooltip:** "The random seed used for creating the noise."
    *   **Description:** The seed for the random number generator.
    *   **Practical Effect:** Ensures reproducibility. Using the same seed with the same parameters will produce the same image. Changing the seed generates a different image.

3.  **`steps`**:
    *   **Type:** `INT`
    *   **Default:** `20`
    *   **Min:** `1`
    *   **Max:** `10000`
    *   **Tooltip:** "The number of steps used in the denoising process."
    *   **Description:** The number of iterations the sampler will perform to denoise the latent image.
    *   **Practical Effect:** More steps generally lead to higher quality and more detailed images, but also increase computation time. Too few steps can result in noisy or incomplete images. The optimal number can depend on the sampler and scheduler.

4.  **`cfg`**:
    *   **Type:** `FLOAT`
    *   **Default:** `8.0`
    *   **Min:** `0.0`
    *   **Max:** `100.0`
    *   **Step:** `0.1`
    *   **Round:** `0.01`
    *   **Tooltip:** "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."
    *   **Description:** Standard Classifier-Free Guidance scale.
    *   **Practical Effect:** Controls adherence to the `positive` prompt. Works in conjunction with NAG parameters to guide the image generation.

5.  **`nag_scale`**:
    *   **Type:** `FLOAT`
    *   **Default:** `5.0`
    *   **Min:** `0.0`
    *   **Max:** `100.0`
    *   **Step:** `0.1`
    *   **Round:** `0.01`
    *   **Description:** Main scaling factor for Negative Guidance.
    *   **Practical Effect:** Directly controls the strength of the `nag_negative` prompt's influence within this sampler. See `NAGCFGGuider.nag_scale` for more details.

6.  **`nag_tau`**:
    *   **Type:** `FLOAT`
    *   **Default:** `2.5`
    *   **Min:** `1.0`
    *   **Max:** `10.0`
    *   **Step:** `0.1`
    *   **Round:** `0.01`
    *   **Description:** Tau parameter for Negative Guidance.
    *   **Practical Effect:** Fine-tunes the NAG behavior, potentially affecting sensitivity or thresholding. See `NAGCFGGuider.nag_tau` for more details.

7.  **`nag_alpha`**:
    *   **Type:** `FLOAT`
    *   **Default:** `0.25`
    *   **Min:** `0.0`
    *   **Max:** `1.0`
    *   **Step:** `0.01`
    *   **Round:** `0.01`
    *   **Description:** Alpha parameter for Negative Guidance.
    *   **Practical Effect:** Modulates the intensity or blending of the NAG correction. See `NAGCFGGuider.nag_alpha` for more details.

8.  **`nag_sigma_end`**:
    *   **Type:** `FLOAT`
    *   **Default:** `0.0`
    *   **Min:** `0.0`
    *   **Max:** `20.0`
    *   **Step:** `0.01`
    *   **Round:** `0.01`
    *   **Description:** Sigma value at which NAG effect ceases.
    *   **Practical Effect:** Determines how long NAG remains active during the denoising process. See `NAGCFGGuider.nag_sigma_end` for more details.

9.  **`sampler_name`**:
    *   **Type:** `KSampler.SAMPLERS` (Dropdown list of available samplers)
    *   **Tooltip:** "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."
    *   **Description:** The specific sampling algorithm (e.g., `euler`, `dpmpp_2m_sde`, `lcm`).
    *   **Practical Effect:** Different samplers can produce vastly different results in terms of image style, detail, coherence, and artifact presence. Some are faster, while others might be better for specific types of images or prompts. The chosen sampler will interact with the NAG parameters.

10. **`scheduler`**:
    *   **Type:** `KSampler.SCHEDULERS` (Dropdown list of available schedulers)
    *   **Tooltip:** "The scheduler controls how noise is gradually removed to form the image."
    *   **Description:** Determines the sequence of noise levels (sigmas) used during the diffusion steps. Examples include `karras`, `sgm_uniform`, `simple`.
    *   **Practical Effect:** The scheduler significantly impacts how the image evolves during denoising. Some schedulers are better suited for certain samplers or for achieving specific effects (e.g., faster convergence, better detail preservation). It works hand-in-hand with the `steps` and `sampler_name`.

11. **`positive`**:
    *   **Type:** `CONDITIONING`
    *   **Tooltip:** "The conditioning describing the attributes you want to include in the image."
    *   **Description:** The positive prompt.
    *   **Practical Effect:** Guides generation towards desired attributes.

12. **`negative`**:
    *   **Type:** `CONDITIONING`
    *   **Tooltip:** "The conditioning describing the attributes you want to exclude from the image."
    *   **Description:** The standard negative prompt.
    *   **Practical Effect:** Steers generation away from undesired attributes using traditional CFG.

13. **`nag_negative`**:
    *   **Type:** `CONDITIONING`
    *   **Tooltip:** "The conditioning describing the attributes you want to exclude from the image for NAG."
    *   **Description:** The negative prompt specifically for the NAG mechanism.
    *   **Practical Effect:** Provides a secondary, often more targeted, way to suppress undesired features, working alongside the `nag_scale`, `nag_tau`, `nag_alpha`, and `nag_sigma_end` parameters.

14. **`latent_image`**:
    *   **Type:** `LATENT`
    *   **Tooltip:** "The latent image to denoise."
    *   **Description:** The initial latent representation, which can be an empty latent (for text-to-image) or a processed latent from a previous step (for image-to-image).
    *   **Practical Effect:** Provides the starting point for the denoising process. Its dimensions influence the output image size.

15. **`denoise`**:
    *   **Type:** `FLOAT`
    *   **Default:** `1.0`
    *   **Min:** `0.0`
    *   **Max:** `1.0`
    *   **Step:** `0.01`
    *   **Tooltip:** "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."
    *   **Description:** Controls how much the input `latent_image` is altered. A value of `1.0` means the latent is completely redrawn based on the prompts. Lower values preserve more of the original latent structure.
    *   **Practical Effect:** Essential for image-to-image tasks. A low denoise value allows for subtle changes or style transfers, while a high value allows for more significant transformations.

### Output:

*   **`LATENT`**:
    *   **Tooltip:** "The denoised latent."
    *   **Description:** The final latent image after the sampling process with integrated NAG. This can be decoded by a VAE to produce the final pixel image.

### Integration of NAG:

The `KSamplerWithNAG` node directly incorporates the NAG logic using the provided `nag_` parameters. Unlike workflows where `NAGCFGGuider` prepares a separate guider object for an advanced sampler, this node handles both the guidance setup and the sampling itself. The `common_ksampler_with_nag` function (called by this node's `sample` method) is responsible for applying these parameters to the model and sampling loop.

The standard sampling parameters (`seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise`) behave as they would in a regular KSampler, but their effects are now modulated and refined by the active NAG parameters. For instance, the `cfg` value still controls the strength of the main positive prompt, while `nag_scale` independently controls the strength of the NAG negative prompt. This dual control allows for a rich interplay in guiding the image generation process.
---
