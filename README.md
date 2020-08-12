# CRAFT - Character-Region Awareness For Text detection

**Đã xong**  
- CRAFT nhẹ hơn so với model gốc (model gốc dùng detect scene text nên cần nhiều filter)  

**Chưa làm**
- Weakly supervised training 

![sample1](sample_images/1.jpg)  
![sample1 region score map](sample_images/1_mask.jpg)  

![sample2](sample_images/2.jpg)  
![sample2 region score map](sample_images/2_mask.jpg)  

Model architecture trong ảnh minh họa của paper bị vẽ sai  
(xem thêm: https://github.com/clovaai/CRAFT-pytorch/issues/24)

Dưới đây là model đúng  
![Correct model architecture](sample_images/correct_original_model.png)