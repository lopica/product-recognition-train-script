# Sử dụng hình ảnh Node.js v18
FROM node:18

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép package.json vào container
COPY package.json ./

# Cài đặt TensorFlow.js v3.18.0
RUN npm install @tensorflow/tfjs-node@3.18.0

# Sao chép mã nguồn vào container
COPY . .

# Chạy ứng dụng
CMD ["node", "server.js"]  # Thay đổi 'index.js' nếu file khởi động của bạn khác
