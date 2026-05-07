# FlowHC: Flow Matching + Hyperbolic Composition for Zero-Shot Compositional Learning
This repository implements a zero-shot compositional learning model based on flow matching and hyperbolic composition modeling.
The method uses flow matching to improve verb-object feature interaction and hyperbolic projection to model compositional structure in representation space.
This repository contains the implementation and training/evaluation scripts for running the FlowHC model.

![thumbnail](assets/thumbnail.png)


## Getting Started
- To prepare the Sth-Com dataset including word embeddings, we refer you to follow the instructions described in the [Sth-Com](https://github.com/RongchangLi/ZSCAR_C2C).
- Most configurations are managed by YAML files under **config/**. Make sure to set appropriate dataset and save directory paths before training.


## Training
```bash
bash script/run_train.sh
```

## Evaluation
```bash
bash script/run_eval.sh
```
Pre-trained checkpoints are available at [this link](https://huggingface.co/heeseokjung/zs-comp-mcr2/tree/main).

## Notes
- For the first few runs, loading the dataset and training might be significantly slow, but it will become much faster in subsequent runs due to OS-level caching. 
- We refactored the codebase to support Distributed Data Parallel (DDP) training for scalability; however, in most cases, single-GPU training showed slightly better performance. All results reported in the paper were obtained using a single GPU setup.
- Flow matching and hyperbolic loss weights can be configured in the YAML files under **config/**.

## Acknowledgements
This repository is built upon [C2C](https://github.com/RongchangLi/ZSCAR_C2C). We sincerely appreciate their efforts in releasing both the codebase and the dataset.


## Citation
If you find this repository useful, please consider citing:
```bibtex
@inproceedings{jung2025zero,
  title={Zero-Shot Compositional Video Learning with Coding Rate Reduction},
  author={Jung, Heeseok and Bak, Jun-Hyeon and Jeong, Yujin and Lee, Gyugeun and Ahn, Jinwoo and Kim, Eun-Sol},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20508--20518},
  year={2025}
}
```
