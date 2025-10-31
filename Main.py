# 🌡️ مشروع التنبؤ بدرجة الحرارة باستخدام LSTM في PyTorch
# -----------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1️⃣ إنشاء بيانات تجريبية (بديلة عن بيانات حقيقية)
np.random.seed(0)
dates = pd.date_range("2020-01-01", periods=500)
temperature = 25 + np.sin(np.linspace(0, 50, 500)) * 10 + np.random.randn(500)
humidity = 50 + np.random.randn(500) * 5
co2 = 400 + np.random.randn(500) * 20

df = pd.DataFrame({
    'Date': dates,
    'Temperature': temperature,
    'Humidity': humidity,
    'CO2': co2
})

# 2️⃣ تطبيع البيانات (Normalization)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Temperature', 'Humidity', 'CO2']])

# 3️⃣ تجهيز البيانات لتصبح تسلسلات زمنية
def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length, 0])  # الهدف: درجة الحرارة
    return np.array(X), np.array(Y)

seq_length = 30  # 30 يوم
X, y = create_sequences(scaled_data, seq_length)

# 4️⃣ تقسيم Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 5️⃣ بناء نموذج LSTM
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # آخر خطوة زمنية
        out = self.fc(out)
        return out

input_size = 3   # Temp, Humidity, CO2
hidden_size = 50
num_layers = 2
output_size = 1

model = WeatherLSTM(input_size, hidden_size, num_layers, output_size)

# 6️⃣ دالة الخسارة والمُحسّن
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7️⃣ تدريب النموذج
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# 8️⃣ التقييم على بيانات الاختبار
model.eval()
with torch.no_grad():
    predicted = model(X_test)
    loss = criterion(predicted, y_test)
    print(f"\nTest Loss: {loss.item():.6f}")

# 9️⃣ رسم النتائج
predicted_np = predicted.numpy()
y_test_np = y_test.numpy()

plt.figure(figsize=(10,5))
plt.plot(y_test_np, label='Actual Temperature')
plt.plot(predicted_np, label='Predicted Temperature')
plt.legend()
plt.title('Actual vs Predicted Temperature')
plt.show()

# 🔟 التنبؤ بالأيام القادمة (Forecasting)
def forecast_future(model, data, seq_length, days_ahead):
    model.eval()
    last_seq = data[-seq_length:]
    preds = []
    for _ in range(days_ahead):
        seq_tensor = torch.tensor(last_seq[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            pred = model(seq_tensor).numpy()
        preds.append(pred[0][0])
        # نضيف التوقع الجديد ونحذف أقدم يوم
        new_row = np.array([[pred[0][0], last_seq[-1,1], last_seq[-1,2]]])
        last_seq = np.vstack((last_seq[1:], new_row))
    return np.array(preds)

future_preds = forecast_future(model, scaled_data, seq_length=30, days_ahead=7)
print("\n🌤️ توقعات الأيام القادمة (قيم مُطبّعة):", future_preds)

# عكس التطبيع (استرجاع القيم الحقيقية)
dummy = np.zeros((len(future_preds), 3))
dummy[:,0] = future_preds
future_real = scaler.inverse_transform(dummy)[:,0]
print("🌡️ توقعات درجات الحرارة القادمة:", future_real)
