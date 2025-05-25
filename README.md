# Eedi Reranker

```markdown
This project implements a reranking model for misconceptions in educational data using a listwise approach. The model is trained and evaluated on a dataset of misconceptions, leveraging a pre-trained language model with fine-tuning. 
```
[infer notebook](https://www.kaggle.com/code/nguynhucng/infer)
## Requirements

- Python 3.8+
- Required Python libraries are listed in `requirements.txt`.

## Installation

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   python src/generate_data.py

    # split data into 5 folds
    python src/split_fold.py
    
    # knowledge distillation
    python src/distill.py
    
    # training biencoder model
    python src/train_biencoder.py --fold 4
    python src/train_biencoder.py --fold 3
    python src/train_biencoder.py --fold 2
    python src/train_biencoder.py --fold 1
    python src/train_biencoder.py --fold 0
    
    # run biencoder model and get topk ids
    python src/run_biencoder.py
    
    # training reranking model
    python src/train_reranking.py --fold 4
    ```

  
## Authors

- **Nguyen Huu Cong**  
  *Email:* cong.nh225476@sis.hust.edu.vn

- **Vu Hoang Nhat Anh**  
  *Email:* anh.vhn20225471@sis.hust.edu.vn

- **Nguyen Duc Anh**  
  *Email:* anh.nd20225468@sis.hust.edu.vn

- **Do Tran Gia Bach**  
  *Email:* bach.dtg225472@sis.hust.edu.vn

- **Tran Thanh Vinh**  
  *Email:* vinh.tt225539@sis.hust.edu.vn