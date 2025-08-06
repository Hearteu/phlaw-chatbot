# Use the official Node.js runtime as a parent image
FROM node:20-alpine

WORKDIR /app

# Copy package.json and package-lock.json to leverage Docker cache
COPY frontend/package*.json ./

# Install npm dependencies
RUN npm install

# Copy the rest of the frontend files
COPY frontend/ ./frontend/

# Expose the Next.js server port
EXPOSE 3000

# Start the Next.js development server
CMD ["npm", "run", "dev"]
