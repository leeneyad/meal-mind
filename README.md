# Meal Mind

*Meal Mind* is a smart web application that recommends healthy breakfast meals for children based on their:

- Age  
- Body Mass Index (BMI)  
- Insulin level  

The system predicts the glucose impact of meals using a trained machine learning model and suggests the *top 3 meals* with the lowest sugar effect.

## Features

- User-friendly web interface built with *Flask*
- Interactive form to collect child’s health data
- Backend model using *Random Forest Regressor*
- Custom illustrations and clean CSS styling
- Hosted live on *Render*

## Live Demo  
[*Click here to try it!*](https://meal-mind.onrender.com)

## Tech Stack

- *Python*
- *Flask*
- *Pandas, scikit-learn, joblib*
- *HTML/CSS*
- *Render* (for deployment)

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/leeneyad/meal-mind.git
   cd meal-mind