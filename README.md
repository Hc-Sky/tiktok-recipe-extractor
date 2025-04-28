# TikTok Recipe Extractor

A web application that extracts recipe information from TikTok videos and allows users to download TikTok videos directly.

## Features

- Extract recipe information from TikTok videos, including:
  - Title and summary
  - Ingredients and quantities
  - Preparation steps
  - Cooking and preparation time
  - Nutrition information
  - Health evaluation
- Download TikTok videos directly
- Generate PDF recipe cards
- Customize recipe extraction based on dietary profiles (Weight Loss, Muscle Gain, Vegetarian)

## Project Structure

The project consists of two main parts:
- Frontend: React application
- Backend: FastAPI Python application

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd tiktok-recipe-backend
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```
   python main.py
   ```
   The backend server will start at http://127.0.0.1:8000

### Frontend Setup

1. Navigate to the project root directory

2. Install dependencies:
   ```
   npm install
   ```

3. Run the frontend development server:
   ```
   npm start
   ```
   The frontend will start at http://localhost:3000

## How to Use

### Extracting Recipes

1. Enter a TikTok video URL in the input field
2. (Optional) Select a dietary profile from the dropdown menu
3. Click the "Extract Recipe" button
4. Wait for the recipe to be extracted and displayed

### Downloading TikTok Videos

1. Enter a TikTok video URL in the input field
2. Click the "Download Video" button
3. Wait for the video to be downloaded
4. Click the "Video ready - Click to download" link to download the video to your device

The application uses the tiktok_scraper library to download TikTok videos efficiently and reliably. It also supports downloading from YouTube URLs using pytube, with fallback to yt-dlp if needed.

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
