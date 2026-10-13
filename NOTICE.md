# Research credits and implementation scope

SuperONN implements polynomial nodal operations and fixed or learned nonlocalized receptive fields described in the research below. These underlying mechanisms are prior work. The library provides conventional convolution interfaces, explicit connectivity and boundary contracts, grouped execution, decoder extensions, and framework integration.

## Core method papers

The [README bibliography](README.md#research-and-citation) identifies the
three core papers and explains their role in this implementation.

- S. Kiranyaz, J. Malik, M. Yamac, M. Duman, I. Adalioglu, E. Guldogan, T. Ince, and M. Gabbouj. *Super Neurons*. IEEE Transactions on Emerging Topics in Computational Intelligence **8**(1) (2024), 206–228. DOI: [10.1109/TETCI.2023.3314658](https://doi.org/10.1109/TETCI.2023.3314658).
- S. Kiranyaz, J. Malik, H. Ben Abdallah, T. Ince, A. Iosifidis, and M. Gabbouj. *Self-organized Operational Neural Networks with Generative Neurons*. Neural Networks **140** (2021), 294–308. DOI: [10.1016/j.neunet.2021.02.028](https://doi.org/10.1016/j.neunet.2021.02.028).
- S. Kiranyaz, T. Ince, A. Iosifidis, and M. Gabbouj. *Operational neural networks*. Neural Computing and Applications **32**(11) (2020), 6645–6668. DOI: [10.1007/s00521-020-04780-3](https://doi.org/10.1007/s00521-020-04780-3).

## Additional implementation context

These references provide further methodological and application context for
the implementation; they are separate from the core bibliography above.

- J. Malik, S. Kiranyaz, and M. Gabbouj. *Self-organized operational neural networks for severe image restoration problems*. Neural Networks **135** (2021), 201–211. DOI: [10.1016/j.neunet.2020.12.014](https://doi.org/10.1016/j.neunet.2020.12.014).
- [BRSR-OpGAN](https://arxiv.org/abs/2407.13949), Neural Networks 190 (2025), 107709.
- *Stabilizing higher-order self-organized operational neural networks via adaptive order-aware constraints for ECG-based myocardial infarction detection*, Information Sciences 750 (2026), 123556. DOI: [10.1016/j.ins.2026.123556](https://doi.org/10.1016/j.ins.2026.123556).

For a publication, cite the papers relevant to the mechanisms used and identify the exact library version or commit. Report offset mode, sharing layout, polynomial degree, input range policy, and regularization so experiments can be compared meaningfully.

## Implementation scope

Offsets are global parameters or buffers shared over spatial positions, kernel taps, polynomial degrees, and samples. Full connectivity assigns offsets to connected input/output pairs; shared connectivity ties them across outputs in a group. Fractional interpolation precedes polynomial evaluation.

The transposed operators in both frameworks are explicit decoder extensions on a sparse upsampled lattice. They are not claimed as an exact reproduction of a published transposed operator. The Keras separable composition uses a depthwise operational layer followed by a conventional pointwise convolution.

API correctness and implementation speed are distinct from scientific novelty and application efficacy. Model quality requires task-specific experiments.

## Licensing

The code in this distribution is licensed under MIT. It does not bundle third-party reference repositories or their operators. Research references do not grant permission to redistribute a referenced codebase; those projects retain their own licenses. PyTorch, TensorFlow, Keras, and other dependencies also retain their respective licenses.
