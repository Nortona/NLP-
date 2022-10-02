from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
import numpy
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


class Get_specific_infomation(object):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.intelligence_id = list(df['intelligence_id'][0:100])
        self.intelligence_content = list(df['intelligence_content'][0:100])
        self.res_person_id = []
        self.res_car_id = []
        self.res_person_name = []
        self.res_adress = []
        self.res_company = []
        self.text = ""
        self.df_res = pd.DataFrame()
        self.model = AutoModelForTokenClassification.from_pretrained(
            "./models/uer/roberta-base-finetuned-cluener2020-chinese")
        self.tokenizer = AutoTokenizer.from_pretrained('./models/uer/roberta-base-finetuned-cluener2020-chinese',
                                                       model_max_length=512)

    def person_id_extract(self):
        person_id = re.findall(
            r"([1-9]\d{5}(18|19|([23]\d))\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx])", self.text)
        per_id = ""
        if person_id:
            matrix = numpy.array(person_id)
            for i in matrix[:, 0]:
                per_id = per_id + "".join(tuple(i)) + ','
        return per_id

    def car_ID_extract(self):
        car_search = r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁台琼使领军北南成广沈济空海]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂领学警港澳]{1}(?!\d)'
        all_car_id = re.findall(car_search, self.text)
        car_id = []
        car_id1 = ""
        if all_car_id:
            for i in all_car_id:
                if not i in car_id:
                    car_id.append(i)
            for i in car_id:
                car_id1 = car_id1 + "".join(tuple(i)) + ','  # 将列表转字符串
        return car_id1  # 返回字符串

    def get_info(self):

        i = 1
        for data in self.intelligence_content:
            print(i)
            i = i + 1
            self.text = data
            person_id = self.person_id_extract()
            self.res_person_id.append(person_id)
            # print("person_id: ", person_id)

            car_id = self.car_ID_extract()
            self.res_car_id.append(car_id)
            # print("car_id: ", car_id)

            nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

            ner_results = nlp(self.text)
            address_temp = ""
            company_temp = ""
            name_temp = ""
            flag_name = 0
            flag_add = 0
            flag_com = 0
            for data in ner_results:
                # print("data", data)

                if 'B-' in data['entity']:
                    if not flag_add:
                        address_temp += data['word']
                        flag_add = 1
                    else:
                        address_temp = address_temp + ',' + data['word']
                    # print("address: ", address_temp)
                elif data['entity'] == 'I-address':
                    address_temp += data['word']

                if data['entity'] == 'B-company':
                    if not flag_com:
                        company_temp += data['word']
                        flag_com = 1
                    else:
                        company_temp = company_temp + ',' + data['word']
                    # print("address: ", address_temp)
                elif data['entity'] == 'I-company':
                    company_temp += data['word']

                if data['entity'] == 'B-name':
                    if not flag_name:
                        name_temp += data['word']
                        flag_name = 1
                    else:
                        name_temp = name_temp + ',' + data['word']
                    # print("address: ", address_temp)
                elif data['entity'] == 'I-name':
                    name_temp += data['word']

            self.res_adress.append(address_temp)
            self.res_person_name.append(name_temp)
            self.res_company.append(company_temp)
        res = {"intelligence_id": self.intelligence_id,
               "intelligence_person_id": self.res_person_id,
               "intelligence_car_id": self.res_car_id,
               "intelligence_person_name": self.res_person_name,
               "intelligence_address": self.res_adress,
               "intelligence_company": self.res_company,
               "intelligence_content": self.intelligence_content}
        self.df_res = pd.DataFrame(res)
        print(self.df_res)
        self.df_res.to_csv("res.csv")
        # dict_list = self.df_res.to_dict(orient='records')
        # spark_pg_object.copy_to_db("dw_intelligence_import_information", dict_list)
        return self.df_res


test_path = "test.csv"
GetInfo = Get_specific_infomation(test_path)
GetInfo.get_info()
