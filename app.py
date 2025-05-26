from flask import Flask, request, render_template  # Flask web framework and HTML handling
import joblib  # For loading the ML model saved using joblib
import numpy as np  # For numeric operations like log()

# Initialize Flask app
app = Flask(__name__)

# Load the trained model from file (joblib format)
model = joblib.load('loan_model.joblib')  # Make sure 'model.joblib' is in the same directory

# Route for homepage
@app.route('/')
def home():
    return render_template("index.html")  # Show the home page (index.html)

# Route to handle prediction logic
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # 1. Get form data from the HTML form (as strings)
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employed = request.form['employed']
        credit = float(request.form['credit'])
        area = request.form['area']
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_term = float(request.form['Loan_Amount_Term'])

        # 2. Encode categorical variables manually (as per training)
        male = 1 if gender == "Male" else 0
        married_yes = 1 if married == "Yes" else 0

        # One-hot encoding for dependents
        dependents_1 = dependents_2 = dependents_3 = 0
        if dependents == '1':
            dependents_1 = 1
        elif dependents == '2':
            dependents_2 = 1
        elif dependents == '3+':
            dependents_3 = 1

        not_graduate = 1 if education == "Not Graduate" else 0
        employed_yes = 1 if employed == "Yes" else 0

        # One-hot encoding for area
        semiurban = urban = 0
        if area == "Semiurban":
            semiurban = 1
        elif area == "Urban":
            urban = 1

        # 3. Validate numerical inputs (avoid divide-by-zero or log(0) errors)
        if applicant_income == 0.0:
            return render_template("prediction.html", prediction_text=" Applicant Income must be greater than 0.")
        if loan_amount == 0.0:
            return render_template("prediction.html", prediction_text=" Loan Amount must be greater than 0.")
        if loan_term == 0.0:
            return render_template("prediction.html", prediction_text="Loan Term must be greater than 0.")

        # 4. Apply log transformation for numerical stability (as used during model training)
        applicant_income_log = np.log(applicant_income)
        total_income_log = np.log(applicant_income + coapplicant_income)
        loan_amount_log = np.log(loan_amount)
        loan_term_log = np.log(loan_term)

        # 5. Arrange the final feature list for model input
        features = [[
            credit,
            applicant_income_log,
            loan_amount_log,
            loan_term_log,
            total_income_log,
            male,
            married_yes,
            dependents_1,
            dependents_2,
            dependents_3,
            not_graduate,
            employed_yes,
            semiurban,
            urban
        ]]

        # 6. Perform prediction using loaded model
        result = model.predict(features)[0]  # Predict returns an array, take the first value

        if(result=="N"):
            result=1
        else:
           result=0
        return render_template("predict.html", prediction_r=result)

    else:
       return render_template("predict.html")


# Route for About Us page
@app.route('/about')
def about():
    return render_template("about.html")

# Run the Flask app in debug mode (for development)
if __name__ == "__main__":
    app.run(debug=True)
