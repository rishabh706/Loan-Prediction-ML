# -*- coding: utf-8 -*-

from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.embed import components

app=Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    # Getting the data from the form
    term=request.form['Term']
    credit_score=request.form['Credit Score']
    annual_income=request.form['Annual Income']
    years_current_job=request.form['Years in current job']
    home_ownership=request.form['Home Ownership']
    purpose=request.form['Purpose']
    monthly_debt=request.form['Monthly Debt']
    years_credit_hist=request.form['Years of Credit History']
    number_of_open_accounts=request.form['Number of Open Accounts']
    number_credit_prob=request.form['Number of Credit Problems']
    max_open_credit=request.form['Maximum Open Credits']
    tax_liens=request.form['Tax Liens']
    current_credit_balance=request.form['Current Credit Balance']
    #  creating a json object to hold the data from the form
    input_data=[{
    'term':term,
    'credit_score':credit_score,
    'annual_income':annual_income,
    'years_current_job':years_current_job,
    'home_ownership':home_ownership,
    'purpose':purpose,
    'monthly_debt':monthly_debt,
    'years_credit_hist':years_credit_hist,
    'number_of_open_accounts':number_of_open_accounts,
    'number_credit_prob':number_credit_prob,
    'current_credit_balance':current_credit_balance,
    'max_open_credit':max_open_credit,
    'tax_liens':tax_liens}]


    dataset=pd.DataFrame(input_data)

    dataset=dataset.rename(columns={
                'term':'Term',
                'credit_score': 'Credit Score',
                'annual_income':'Annual Income',
                'years_current_job':'Years in current job',
                'home_ownership':'Home Ownership',
                'purpose':'Purpose',
                'monthly_debt':'Monthly Debt',
                'years_credit_hist':'Years of Credit History',
                'number_of_open_accounts':'Number of Open Accounts',
                'number_credit_prob':'Number of Credit Problems',
                'current_credit_balance':'Current Credit Balance',
                'max_open_credit':'Maximum Open Credit',
                'tax_liens':'Tax Liens'})

    dataset[['Credit Score','Annual Income','Monthly Debt','Years of Credit History',
             'Number of Open Accounts', 'Number of Credit Problems', 'Current Credit Balance', 'Maximum Open Credit', 'Tax Liens']] = dataset[['Credit Score', 'Annual Income', 'Monthly Debt', 'Years of Credit History', 'Number of Open Accounts', 'Number of Credit Problems', 'Current Credit Balance', 'Maximum Open Credit', 'Tax Liens']].astype(float)

    dataset[['Term','Years in current job','Home Ownership','Purpose']]=dataset[['Term','Years in current job','Home Ownership','Purpose']].astype('object')

    dataset = dataset[['Term','Credit Score','Annual Income','Years in current job',
    'Home Ownership','Purpose','Monthly Debt','Years of Credit History','Number of Open Accounts','Number of Credit Problems','Current Credit Balance','Maximum Open Credit','Tax Liens']]          
    model = pickle.load(open('model.pkl', 'rb'))
    classifier=model.predict_proba(dataset)
    predictions = [item for sublist in classifier for item in sublist]
    colors = ['#1f77b4','#ff7f0e']
    loan_status = ['Charged Off','Fully Paid']
    source = ColumnDataSource(
        data=dict(loan_status=loan_status, predictions=predictions))

    p = figure(x_range=loan_status, plot_height=500,
               toolbar_location=None, title="Loan Status", plot_width=800)
    p.vbar(x='loan_status', top='predictions', width=0.4, source=source, legend="loan_status",
           line_color='black', fill_color=factor_cmap('loan_status', palette=colors, factors=loan_status))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0.1
    p.y_range.end = 0.9
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.axis_label = 'Loan Status'
    p.yaxis.axis_label = ' Predicted Probabilities'
    script, div = components(p)
    return render_template('result.html',script=script,div=div)




if __name__=="__main__":
    app.run(debug=True)
