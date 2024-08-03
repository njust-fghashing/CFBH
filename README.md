# Alleviating Over-fitting in Hashing-based Fine-grained Image Retrieval: From Causal Feature Learning to Binary-injected Hash Learning

Dataset Preparation
---
Move the dataset into the corresponding path ./dataset like the above <br>
<details>
<summary>Details</summary>

```python
|--dataset
  |--cub_bird
    |--images 
         |--001...
         |--002... 
         ... 
    |--classes.txt 
    |--image_class_labels.txt 
    |--image.txt 
    |--train_test_split.txt
    |--cub_bird_test.txt
    |--cub_bird_train.txt
```
</details>




---
Train
---
(1) Put the parameters of Resnet18 into the path ./petrained. This parameters can be download at PyTorch official link：https://download.pytorch.org/models/resnet18-f37072fd.pth. <br>

(2) Train the network, such as: python CFBH.py --dataset cub_bird --ratio 0.25 --num_parts 64 	 <br>

Citation
---
@ARTICLE{10566715,
  author={Xiang, Xinguang and Ding, Xinhao and Jin, Lu and Li, Zechao and Tang, Jinhui and Jain, Ramesh},   
  journal={IEEE Transactions on Multimedia},   
  title={Alleviating Over-fitting in Hashing-based Fine-grained Image Retrieval: From Causal Feature Learning to Binary-injected Hash Learning},     
  year={2024},    
  volume={},    
  number={},      
  pages={1-13},       
  keywords={Hashing-based fine-grained image retrieval;over-fitting;causal inference},      
  doi={10.1109/TMM.2024.3410136}}
