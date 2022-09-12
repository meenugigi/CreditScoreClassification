import pandas as pd
from flask import Flask, render_template, request, app, abort
import pickle
from DataCleaningAndProcessing import removing_nulls, data_encoding, data_cleaning, reverse_encoding


app=Flask(__name__)


pipe = pickle.load(open("RandomForestClassifierModel.pkl",'rb'))


@app.route("/")
def index():

    # data = pd.read_csv('train.csv')
    dataclean1 = removing_nulls()
    cleaned_data = data_cleaning(dataclean1)
    # print(cleaned_data)
    occupation = sorted(cleaned_data['Occupation'].unique())
    credit_mix = sorted(cleaned_data['Credit_Mix'].unique())
    payment_behaviour = sorted(cleaned_data['Payment_Behaviour'].unique())
    payment_min_amount = sorted(cleaned_data['Payment_of_Min_Amount'].unique())
    index.d1, index.d2, index.d3, index.d4 = data_encoding(cleaned_data)
    return render_template('index.html',occupation = occupation, credit_mix = credit_mix,
                           payment_behaviour = payment_behaviour, payment_min_amount=payment_min_amount)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Name = request.form.get('Name')
        Occupation = request.form.get('Occupation')
        Occupation_Decoded = reverse_encoding(Occupation, index.d1)

        Monthly_Inhand_Salary = request.form.get('Monthly_Inhand_Salary')
        Num_Bank_Accounts = request.form.get('Num_Bank_Accounts')
        Num_Credit_Card = request.form.get('Num_Credit_Card')
        Interest_Rate = request.form.get('Interest_Rate')
        Num_of_Loan = request.form.get('Num_of_Loan')

        Delay_from_due_date = request.form.get('Delay_from_due_date')
        Num_of_Delayed_Payment = request.form.get('Num_of_Delayed_Payment')
        Changed_Credit_Limit = request.form.get('Changed_Credit_Limit')
        Num_Credit_Inquiries = request.form.get('Num_Credit_Inquiries')
        Credit_Mix = request.form.get('Credit_Mix')
        print("------------", index.d3)
        Credit_Mix_Decoded = reverse_encoding(Credit_Mix, index.d3)

        Outstanding_Debt = request.form.get('Outstanding_Debt')
        Credit_Utilization_Ratio = request.form.get('Credit_Utilization_Ratio')

        Credit_History_Age = request.form.get('Credit_History_Age')
        Payment_of_Min_Amount = request.form.get('Payment_of_Min_Amount')
        Payment_of_Min_Amount_Decoded = reverse_encoding(Payment_of_Min_Amount, index.d4)

        Total_EMI_per_month = request.form.get('Total_EMI_per_month')
        Amount_invested_monthly = request.form.get('Amount_invested_monthly')
        Payment_Behaviour = request.form.get('Payment_Behaviour')
        Payment_Behaviour_Decoded = reverse_encoding(Payment_Behaviour, index.d2)
        Monthly_Balance = request.form.get('Monthly_Balance')

        # print(Name,Occupation, Monthly_Inhand_Salary, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate,
        #                        Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit,
        #                        Num_Credit_Inquiries, Credit_Mix, Outstanding_Debt,Credit_Utilization_Ratio, Credit_History_Age,
        #                        Payment_of_Min_Amount,Total_EMI_per_month,Amount_invested_monthly,Payment_Behaviour,Monthly_Balance)

        input = pd.DataFrame([[Num_Bank_Accounts, Num_Credit_Card, Interest_Rate,
                               Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit,
                               Num_Credit_Inquiries, Credit_Mix_Decoded, Outstanding_Debt,Credit_Utilization_Ratio, Credit_History_Age,
                               Payment_of_Min_Amount_Decoded,Total_EMI_per_month,Amount_invested_monthly,Payment_Behaviour_Decoded,Monthly_Balance]],
                             columns=['Num_Bank_Accounts', 'Num_Credit_Card',
                                      'Interest_Rate','Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                                      'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt','Credit_Utilization_Ratio',
                                      'Credit_History_Age', 'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly',
                                      'Payment_Behaviour', 'Monthly_Balance'])
        prediction = pipe.predict(input)[0]
        result = prediction
        return str(result)
    except ValueError:
        abort(400, 'Cannot have blank fields')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True, port=5000)
