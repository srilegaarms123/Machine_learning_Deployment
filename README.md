
# 🌍 Country Prediction Using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based web application that predicts a customer's country using a trained Random Forest model.

Users can enter the following details through a simple web form:

- Quantity
- Unit Price
- Customer ID

After submitting the details, the application predicts the customer's country and displays the result on the webpage.

The project uses Flask to connect the Machine Learning model with a web interface and Render to deploy the application online.

---

## 🎯 Project Objective

The main objective of this project is to understand how a trained Machine Learning model can be deployed as a real-world web application.

This project demonstrates the complete Machine Learning deployment workflow:

1. Train a Machine Learning model.
2. Save the trained model as a Joblib file.
3. Build a Flask web application.
4. Load the saved model into Flask.
5. Accept user input through a web form.
6. Generate predictions using the model.
7. Display the prediction to the user.
8. Deploy the application using Render.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data handling and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning model |
| Random Forest | Country prediction algorithm |
| Joblib | Saving and loading the trained model |
| Flask | Building the web application |
| HTML | Creating the user input form |
| CSS | Styling the webpage |
| JavaScript | Sending input to Flask and displaying results |
| GitHub | Source code and model storage |
| Render | Hosting and deploying the application |

---

## 🧠 Machine Learning Algorithm

This project uses the **Random Forest Classifier**.

Random Forest is a supervised Machine Learning algorithm that combines multiple decision trees to make a prediction.

The trained model uses these input features:

- Quantity
- UnitPrice
- CustomerID

The target variable is:

- Country

The model learns patterns from the training dataset and predicts the country based on the values entered by the user.

---

## 📂 Project Folder Structure

```text
country-prediction-project/
│
├── app.py
│
├── random_forest_model_com.joblib
│
├── requirements.txt
│
└── README.md
```

### File Description

**1. app.py**

Contains the Flask application.

Responsibilities:
- Loads the trained model.
- Displays the input form.
- Receives user input.
- Sends input features to the model.
- Returns the predicted country.
- Displays the prediction result.

**2. random_forest_model_com.joblib**

Contains the saved Machine Learning model and supporting information.

It stores:
- Trained Random Forest model.
- Target encoder.
- Feature names.

**3. requirements.txt**

Contains the Python libraries required to run the project.

**4. README.md**

Explains the project, setup instructions, and workflow.

---

## ⚙️ Project Workflow

### Step 1: Model Training

First, prepare the dataset and train the Random Forest Classifier using the selected input features.

The model learns the relationship between customer transaction information and the country.

### Step 2: Save the Trained Model

After training, save the model using Joblib.

The saved file contains the model, target encoder, and feature names.

This avoids training the model again whenever the application starts.

### Step 3: Flask Application

Create a Flask application using `app.py`.

Flask loads the saved Joblib file and makes the trained model available for predictions.

### Step 4: User Input

The webpage provides three input fields:

- Quantity
- UnitPrice
- CustomerID

The user enters the values and clicks the Predict button.

### Step 5: Prediction

JavaScript sends the entered values to the Flask `/predict` endpoint.

Flask:
1. Receives the input data.
2. Creates a DataFrame with the expected feature names and order.
3. Passes the data to the trained model.
4. Predicts the country.
5. Converts the predicted encoded value back to the country name using the target encoder.
6. Returns the prediction result.

### Step 6: Display Result

The predicted country is displayed on the webpage.

If the model supports probability prediction, the application can also display the prediction confidence.

### Step 7: Deployment

The project is uploaded to GitHub and connected to Render.

Render installs the required libraries and starts the Flask application using Gunicorn.

The application becomes accessible through a public URL.

---

## 🔄 Complete Project Architecture

```text
          User
           |
           v
     Web Input Form
           |
           v
       JavaScript
           |
           v
     Flask Application
           |
           v
     Load Joblib Model
           |
           v
    Random Forest Model
           |
           v
     Predict Country
           |
           v
      Target Encoder
           |
           v
    Predicted Country
           |
           v
      Display Result
```

---

## 💻 Installation and Setup

### 1. Clone the GitHub Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd country-prediction-project
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 5. Run the Flask Application

```bash
python app.py
```

### 6. Open the Application

Open this URL in your browser:

```text
http://127.0.0.1:5000/
```

Enter the input values and click the Predict button to see the predicted country.

---

## 📦 Requirements

The `requirements.txt` file should contain the required libraries:

```text
Flask
gunicorn
joblib
pandas
numpy
scikit-learn
```

For reliable deployment, use package versions compatible with the environment in which the model was trained.

---

## ☁️ Deployment Using Render

1. Upload the project files to a GitHub repository.
2. Log in to [Render](https://render.com/).
3. Create a new Web Service.
4. Connect your GitHub repository.
5. Select Python as the environment.
6. Set the Build Command:

```bash
pip install -r requirements.txt
```

7. Set the Start Command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

8. Click Deploy.

After successful deployment, Render provides a public URL to access the application.

---

## 🌐 Live Application

**Live Demo:**  
https://machine-learning-deployment-719e.onrender.com/

Open the link in a browser to access the deployed application.

---

## 📊 Expected Input and Output

### Input Example

| Feature | Example Value |
|---|---:|
| Quantity | 5 |
| UnitPrice | 10.50 |
| CustomerID | 17850 |

### Output Example

```text
Predicted Country: United Kingdom
```

*Note: This is an illustrative example. Actual predictions depend on the trained model and input values.*

---

## 📚 What We Learn From This Project

By completing this project, beginners can understand:

- How to train a Machine Learning classification model.
- How to save and load a model using Joblib.
- How to create a web application using Flask.
- How to connect HTML forms with a Flask backend.
- How to send data using JavaScript Fetch API.
- How to generate predictions from user input.
- How to upload a project to GitHub.
- How to deploy a Machine Learning application using Render.

---

## 🚀 Future Improvements

- Improve model accuracy through hyperparameter tuning.
- Add more useful input features.
- Improve the user interface.
- Add prediction history.
- Display additional prediction information.
- Add better input validation and error handling.

---

## 👨‍💻 Conclusion

This project demonstrates how a Machine Learning model can be converted into a practical web application.

Instead of running predictions only inside a Python notebook, users can interact with the trained Random Forest model through a browser.

It provides a beginner-friendly example of the complete Machine Learning workflow, from model training to real-world deployment.
