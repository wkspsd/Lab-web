from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# 加載模型
model = tf.keras.models.load_model("your_model_directory/your_model.h5")

# 定義 API 端點，接受上傳的圖片
@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        # 讀取上傳的圖片
        image = await file.read()
        image = Image.open(io.BytesIO(image))
        image = image.resize((224, 224))  # 根據模型的輸入大小調整圖片大小

        # 對圖片進行預處理，取決於模型的要求
        # 例如，可能需要對圖片進行歸一化或轉換為 NumPy 陣列

        # 進行模型預測
        # 假設模型能夠接受 NumPy 陣列作為輸入
        prediction = model.predict(image)

        # 根據模型的預測結果構建響應
        # 取決於模型輸出和應用的需求
        response_data = {"prediction": prediction.tolist()}

        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        error_message = {"error": str(e)}
        return JSONResponse(content=error_message, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
