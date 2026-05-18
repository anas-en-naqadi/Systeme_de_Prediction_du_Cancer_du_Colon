# Multi-stage build for React/Vite frontend

FROM node:20-alpine AS builder
WORKDIR /usr/src/app
COPY package.json package-lock.json* ./
COPY ./src ./src
COPY index.html ./
COPY vite.config.js ./
RUN npm ci --silent
RUN npm run build

# Serve static with a lightweight Node server
FROM node:20-alpine
WORKDIR /usr/src/app
RUN npm install -g serve@14.2.0 --silent
COPY --from=builder /usr/src/app/dist ./dist
EXPOSE 5173
CMD ["serve", "-s", "dist", "-l", "5173"]
