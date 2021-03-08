import pandas as pd

excel=pd.DataFrame()
chatbot_data = pd.read_csv('./datasets/ChatbotData .csv', encoding='utf-8')
excel['Q']=chatbot_data['Q']
excel['A']=chatbot_data['A']

excel1=excel[:100]
excel1.to_excel('ChatbotData_1.xlsx', sheet_name = "Sheet_1")

excel2=excel[101:200]
excel2.to_excel('ChatbotData_2.xlsx', sheet_name = "Sheet_1")

excel3=excel[201:300]
excel3.to_excel('ChatbotData_3.xlsx', sheet_name = "Sheet_1")

excel4=excel[301:400]
excel4.to_excel('ChatbotData_4.xlsx', sheet_name = "Sheet_1")

excel5=excel[401:500]
excel5.to_excel('ChatbotData_5.xlsx', sheet_name = "Sheet_1")