from flask import Flask, render_template, request
import pandas as pd
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import keras
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform as U
app = Flask(__name__)

model = keras.models.load_model('model')

ad_inventory=['Elementary', 'Middle', 'HS_grad', 'Undergraduate', 'Graduate'] 

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

def generate_user(df_data):
    user = df_data.sample(1)
    context = user.iloc[:, :-1].values.tolist()[0]
    return user.to_dict(orient='records')[0], context

def get_ad_inventory():
    ad_inv_prob = {'Elementary': 0.9, 
                   'Middle':  0.7, 
                   'HS-grad':  0.7, 
                   'Undergraduate':  0.7, 
                   'Graduate':  0.8}
    ad_inventory = []
    for level, prob in ad_inv_prob.items():
        if U() < prob:
            ad_inventory.append(level)
    # Make sure there are at least one ad among all 
    if not ad_inventory:
        ad_inventory = get_ad_inventory()
    return ad_inventory

def ad_to_one_hot(ad):
    ed_levels = ['Elementary', 
                 'Middle', 
                 'HS-grad', 
                 'Undergraduate', 
                 'Graduate']
    ad_input = [0] * len(ed_levels)
    if ad in ed_levels:
        ad_input[ed_levels.index(ad)] = 1
    return ad_input

def get_ad_click_probs():
    base_prob = 0.8 # When an ad is shown to the user and if ad's target matches then probability of clicking on an ad
    delta = 0.4     # Probability decrease by 0.3 for each level of mismatch
    ed_levels = {'Elementary': 1, 
                 'Middle':  2, 
                 'HS-grad':  3, 
                 'Undergraduate':  4, 
                 'Graduate':  5}
    ad_click_probs = {l1: {l2: max(0, base_prob - delta * abs(ed_levels[l1]- ed_levels[l2])) for l2 in ed_levels}
                           for l1 in ed_levels}
    return ad_click_probs

def select_ad(model, context, ad_inventory):
    selected_ad = None
    selected_x = None
    max_action_val = 0
    for ad in ad_inventory:
        ad_x = ad_to_one_hot(ad)
        x = np.array(context + ad_x).reshape((1, -1))
        action_val_pred = model.predict(x)[0][0]
        if action_val_pred >= max_action_val:
            selected_ad = ad
            selected_x = x
            max_action_val = action_val_pred
    return selected_ad, selected_x

def display_ad(ad_click_probs, user, ad):
    prob = ad_click_probs[ad][user['education']]
    click = 1 if U() < prob else 0 #U() Generate a random float number between 0 to 1
    return click

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age']  )    
        education=request.form['education']
        marital_status=request.form['marital_status']
        occupation=request.form['occupation']
        gender=request.form['gender']
        hours_per_week=int(request.form['hours_per_week'])
        native_country=request.form['native_country']
        
        income=request.form['income']
        if(income=='incomeless50K'):
            incomeless50K=1
            incomemore50K=0            
            
        if(income=='incomemore50K'):
            incomeless50K=0
            incomemore50K=1
            
        workclass=request.form['workclass']
        if(workclass=='State_gov'):
                workclass_Federal_gov=0
                workclass_Local_gov=0
                workclass_Private=0
                workclass_Self_emp_inc=0
                workclass_Self_emp_not_inc=0
                workclass_State_gov=1
                workclass_Without_pay=0
                
        if(workclass=='Self_emp_not_inc'):
                workclass_Federal_gov=0
                workclass_Local_gov=0
                workclass_Private=0
                workclass_Self_emp_inc=0
                workclass_Self_emp_not_inc=1
                workclass_State_gov=0
                workclass_Without_pay=0
                
        if(workclass=='Private'):
                workclass_Federal_gov=0
                workclass_Local_gov=0
                workclass_Private=1
                workclass_Self_emp_inc=0
                workclass_Self_emp_not_inc=0
                workclass_State_gov=0
                workclass_Without_pay=0
                
        if(workclass=='Federal_gov'):
                workclass_Federal_gov=1
                workclass_Local_gov=0
                workclass_Private=0
                workclass_Self_emp_inc=0
                workclass_Self_emp_not_inc=0
                workclass_State_gov=0
                workclass_Without_pay=0
                
        if(workclass=='Local_gov'):
                workclass_Federal_gov=0
                workclass_Local_gov=1
                workclass_Private=0
                workclass_Self_emp_inc=0
                workclass_Self_emp_not_inc=0
                workclass_State_gov=1
                workclass_Without_pay=0
                   
              
        if(workclass=='Without_pay'):
                workclass_Federal_gov=0
                workclass_Local_gov=0
                workclass_Private=0
                workclass_Self_emp_inc=0
                workclass_Self_emp_not_inc=0
                workclass_State_gov=0
                workclass_Without_pay=1    
                
        else:
                workclass_Federal_gov=0
                workclass_Local_gov=0
                workclass_Private=0
                workclass_Self_emp_inc=0
                workclass_Self_emp_not_inc=0
                workclass_State_gov=0
                workclass_Without_pay=0
            
        
        
        
        
        marital_status=request.form['marital_status']
        if(marital_status=='Never_married'):
                marital_status_Divorced=0
                marital_status_Married_AF_spouse=0
                marital_status_Married_civ_spouse=0
                marital_status_Married_spouse_absent=0
                marital_status_Never_married=1
                marital_status_Separated=0
                marital_status_Widowed=0
        
        if(marital_status=='Married_civ_spouse'):
                marital_status_Divorced=0
                marital_status_Married_AF_spouse=0
                marital_status_Married_civ_spouse=1
                marital_status_Married_spouse_absent=0
                marital_status_Never_married=0
                marital_status_Separated=0
                marital_status_Widowed=0
                
        if(marital_status=='Divorced'):
                marital_status_Divorced=1
                marital_status_Married_AF_spouse=0
                marital_status_Married_civ_spouse=0
                marital_status_Married_spouse_absent=0
                marital_status_Never_married=0
                marital_status_Separated=0
                marital_status_Widowed=0
        
                
        if(marital_status=='Married_spouse_absent'):
                marital_status_Divorced=0
                marital_status_Married_AF_spouse=0
                marital_status_Married_civ_spouse=0
                marital_status_Married_spouse_absent=1
                marital_status_Never_married=0
                marital_status_Separated=0
                marital_status_Widowed=0
                
        if(marital_status=='Separated'):
                marital_status_Divorced=0
                marital_status_Married_AF_spouse=0
                marital_status_Married_civ_spouse=0
                marital_status_Married_spouse_absent=0
                marital_status_Never_married=0
                marital_status_Separated=1
                marital_status_Widowed=0
                
        if(marital_status=='Married_AF_spouse'):
                marital_status_Divorced=0
                marital_status_Married_AF_spouse=1
                marital_status_Married_civ_spouse=0
                marital_status_Married_spouse_absent=0
                marital_status_Never_married=0
                marital_status_Separated=0
                marital_status_Widowed=0  
                
        if(marital_status=='Widowed'):
                marital_status_Divorced=0
                marital_status_Married_AF_spouse=0
                marital_status_Married_civ_spouse=0
                marital_status_Married_spouse_absent=0
                marital_status_Never_married=0
                marital_status_Separated=0
                marital_status_Widowed=1
                
        else:
                marital_status_Divorced=0
                marital_status_Married_AF_spouse=0
                marital_status_Married_civ_spouse=0
                marital_status_Married_spouse_absent=0
                marital_status_Never_married=0
                marital_status_Separated=0
                marital_status_Widowed=0
         
            
        occupation=request.form['occupation']
        if(occupation=='Adm_clerical'):
               occupation_Adm_clerical=1
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0
               
        if(occupation=='Exec_managerial'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=1
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0      
          
        if(occupation=='Handlers_cleaners'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=1
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0      
           
        if(occupation=='Prof_specialty'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=1
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0
               
        if(occupation=='Other_service'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=1
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0      
               
        if(occupation=='Sales'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=1
               occupation_Tech_support=0
               occupation_Transport_moving=0
               
               
        if(occupation=='Transport_moving'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=1    
               
        if(occupation=='Farming_fishing'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=1
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0
               
        if(occupation=='Machine_op_inspct'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=1
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0     
               
               
        if(occupation=='Tech_support'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=1
               occupation_Transport_moving=0
               
        if(occupation=='Craft_repair'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=1
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0    
               
        if(occupation=='Protective_serv'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=1
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0   
             
               
        if(occupation=='Armed_Forces'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=1
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0 
               
               
        if(occupation=='Priv_house_serv'):
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=1
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0     
            
               
        else:
               occupation_Adm_clerical=0
               occupation_Armed_Forces=0
               occupation_Craft_repair=0
               occupation_Exec_managerial=0
               occupation_Farming_fishing=0
               occupation_Handlers_cleaners=0
               occupation_Machine_op_inspct=0
               occupation_Other_service=0
               occupation_Priv_house_serv=0
               occupation_Prof_specialty=0
               occupation_Protective_serv=0
               occupation_Sales=0
               occupation_Tech_support=0
               occupation_Transport_moving=0 
               
        gender=request.form['gender']
        if(gender=='Female'):
            gender_Female=1
            gender_Male=0
        else:
            gender_Female=0
            gender_Male=1
        
        income=request.form['income']
        if(income=='<=50K'):
            incomeless50K=0
            incomemore50K=1
        else:
            incomeless50K=1
            incomemore50K=0
       
        native_country=request.form['native_country']
        if(native_country=='Cambodia'):
            native_country_Cambodia=1
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0
        
        if(native_country=='Canada'):
            native_country_Cambodia=0
            native_country_Canada=1
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0
        
        if(native_country=='China'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0
       
        
        if(native_country=='Columbia'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=1
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0
       
        if(native_country=='Cuba'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=1
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0
       
        if(native_country=='Dominican_Republic'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=1
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0
            
        if(native_country=='Ecuador'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=1
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0   
            
        if(native_country=='El_Salvador'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=1
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0   
            
            
        if(native_country=='England'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=1
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
            
        if(native_country=='France'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=1
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
            
        if(native_country=='Germany'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=1
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
            
        if(native_country=='Greece'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=1
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0   
            
        if(native_country=='Guatemala'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=1
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
            
        if(native_country=='Haiti'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=1
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
            
        if(native_country=='Holand_Netherlands'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=1
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0   
            
            
        if(native_country=='Honduras'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=1
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
            
            
        if(native_country=='Hong'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=1
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
            
            
        if(native_country=='Hungary'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=1
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0   
            
        if(native_country=='India'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=1
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
            
            
        if(native_country=='Iran'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=1
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0     
        if(native_country=='Ireland'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=1
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
            
        if(native_country=='Italy'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=1
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0    
        if(native_country=='Jamaica'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=1
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Japan'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=1
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        
        if(native_country=='Laos'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=1
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Mexico'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=1
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Nicaragua'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=1
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Outlying_US'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=1
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Peru'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=1
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Philippines'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=1
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Poland'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=1
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Portugal'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=1
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Puerto_Rico'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=1
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0 
            
        if(native_country=='Scotland'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=1
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='South'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=1
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='Taiwan'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=1
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0 
            
        if(native_country=='Trinadad_Tobago'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=1
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
        if(native_country=='United_States'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=1
            native_country_Vietnam=0
            native_country_Yugoslavia=0  
            
        if(native_country=='Vietnam'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=1
            native_country_Yugoslavia=0  
        if(native_country=='Yugoslavia'):
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=1 
        else:
            native_country_Cambodia=0
            native_country_Canada=0
            native_country_China=0
            native_country_Columbia=0
            native_country_Cuba=0
            native_country_Dominican_Republic=0
            native_country_Ecuador=0
            native_country_El_Salvador=0
            native_country_England=0
            native_country_France=0
            native_country_Germany=0
            native_country_Greece=0
            native_country_Guatemala=0
            native_country_Haiti=0
            native_country_Holand_Netherlands=0
            native_country_Honduras=0
            native_country_Hong=0
            native_country_Hungary=0
            native_country_India=0
            native_country_Iran=0
            native_country_Ireland=0
            native_country_Italy=0
            native_country_Jamaica=0
            native_country_Japan=0
            native_country_Laos=0
            native_country_Mexico=0
            native_country_Nicaragua=0
            native_country_Outlying_US=0
            native_country_Peru=0
            native_country_Philippines=0
            native_country_Poland=0
            native_country_Portugal=0
            native_country_Puerto_Rico=0
            native_country_Scotland=0
            native_country_South=0
            native_country_Taiwan=0
            native_country_Thailand=0
            native_country_Trinadad_Tobago=0
            native_country_United_States=0
            native_country_Vietnam=0
            native_country_Yugoslavia=0
            
            
        d = {'age': [age],
         'hours_per_week': [hours_per_week],
         'workclass_Federal_gov': [workclass_Federal_gov],
         'workclass_Local_gov': [workclass_Local_gov],
         'workclass_Private': [workclass_Private],
         'workclass_Self_emp_inc': [workclass_Self_emp_inc],
         'workclass_Self_emp_not_inc': [workclass_Self_emp_not_inc],
         'workclass_State_gov': [workclass_State_gov],
         'workclass_Without_pay': [workclass_Without_pay],
         'marital_status_Divorced': [marital_status_Divorced],
         'marital_status_Married_AF_spouse': [marital_status_Married_AF_spouse],
         'marital_status_Married_civ_spouse': [marital_status_Married_civ_spouse],
         'marital_status_Married_spouse_absent': [marital_status_Married_spouse_absent],
         'marital_status_Never_married': [marital_status_Never_married],
         'marital_status_Separated': [marital_status_Separated],
         'marital_status_Widowed': [marital_status_Widowed],
         'occupation_Adm_clerical': [occupation_Adm_clerical],
         'occupation_Armed_Forces': [occupation_Armed_Forces],
         'occupation_Craft_repair': [occupation_Craft_repair],
         'occupation_Exec_managerial': [occupation_Exec_managerial],
         'occupation_Farming_fishing': [occupation_Farming_fishing],
         'occupation_Handlers_cleaners': [occupation_Handlers_cleaners],
         'occupation_Machine_op_inspct': [occupation_Machine_op_inspct],
         'occupation_Other_service': [occupation_Other_service],
         'occupation_Priv_house_serv': [occupation_Priv_house_serv],
         'occupation_Prof_specialty': [occupation_Prof_specialty],
         'occupation_Protective_serv': [occupation_Protective_serv],
         'occupation_Sales': [occupation_Sales],
         'occupation_Tech_support': [occupation_Tech_support],
         'occupation_Transport_moving': [occupation_Transport_moving],
         'gender_Female': [gender_Female],
         'gender_Male': [gender_Male],
         'native_country_Cambodia': [native_country_Cambodia],
         'native_country_Canada': [native_country_Canada],
         'native_country_China': [native_country_China],
         'native_country_Columbia': [native_country_Columbia],
         'native_country_Cuba': [native_country_Cuba],
         'native_country_Dominican_Republic': [native_country_Dominican_Republic],
         'native_country_Ecuador': [native_country_Ecuador],
         'native_country_El_Salvador': [native_country_El_Salvador],
         'native_country_England': [native_country_England],
         'native_country_France': [native_country_France],
         'native_country_Germany': [native_country_Germany],
         'native_country_Greece': [native_country_Greece],
         'native_country_Guatemala': [native_country_Guatemala],
         'native_country_Haiti': [native_country_Haiti],
         'native_country_Holand_Netherlands': [native_country_Holand_Netherlands],
         'native_country_Honduras': [native_country_Honduras],
         'native_country_Hong': [native_country_Hong],
         'native_country_Hungary': [native_country_Hungary],
         'native_country_India': [native_country_India],
         'native_country_Iran': [native_country_Iran],
         'native_country_Ireland': [native_country_Ireland],
         'native_country_Italy': [native_country_Italy],
         'native_country_Jamaica': [native_country_Jamaica],
         'native_country_Japan': [native_country_Japan],
         'native_country_Laos': [native_country_Laos],
         'native_country_Mexico': [native_country_Mexico],
         'native_country_Nicaragua': [native_country_Nicaragua],
         'native_country_Outlying_US': [native_country_Outlying_US],
         'native_country_Peru': [native_country_Peru],
         'native_country_Philippines': [native_country_Philippines],
         'native_country_Poland': [native_country_Poland],
         'native_country_Portugal': [native_country_Portugal],
         'native_country_Puerto_Rico': [native_country_Puerto_Rico],
         'native_country_Scotland': [native_country_Scotland],
         'native_country_South': [native_country_South],
         'native_country_Taiwan': [native_country_Taiwan],
         'native_country_Thailand': [native_country_Thailand],
         'native_country_Trinadad_Tobago': [native_country_Trinadad_Tobago],
         'native_country_United_States': [native_country_United_States],
         'native_country_Vietnam': [native_country_Vietnam],
         'native_country_Yugoslavia': [native_country_Yugoslavia],
         'incomeless50K': [incomeless50K],
         'incomemore50K': [incomemore50K],
         'education': education}
        
        df = pd.DataFrame(data=d) 
        ad_click_probs = get_ad_click_probs()
        user, context = generate_user(df)
        ad_inventory = get_ad_inventory()
        ad, x = select_ad(model, context, ad_inventory)
        click = display_ad(ad_click_probs, user, ad)

        if click==1:
            action="CLICKED"
        else:
            action="NOT CLICKED"
        
        return render_template('index.html',prediction_text="Predicted degree {} but user is {} action {}".format(ad,user['education'],action))
        
        
        #prediction=model.predict([[Present_Price,Kms_Driven2,Owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Mannual]])
        #output=round(prediction[0],2)
        #if output<0:
        #    return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        #else:
        #     return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

