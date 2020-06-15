# Vietnamese-Text-Classification
Vietnamese Text Classification

***tf-idf, bow, Random Forest,K Nearest Neighbors, Logistic Regression, Support Vector Machine,..***

## Chương trình phân loại 
- Chạy chương trình phân loại 1 văn bản bất kì (model đã tồn tại trong train_model): 
```
python predict.py -i <filename document>
```
------
## Build lại chương trình 

- Tạo thư mục dataset và features : 
```
./mkdir.sh
```
#### Xây dựng dữ liệu & Tiền xử lý
```
python preProcessData.py
```
- Dữ liệu tiền xử lý và lưu trong thư mục  processsed_data 
- Trích chọn đặc trưng đưa vào thư mục feature_extraction

### Chạy chương trình train : 
```
python main.py
```
- File model lưu trong thư mục trained_model
