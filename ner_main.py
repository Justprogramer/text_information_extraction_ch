# -*-coding:utf-8-*-
import os

from ner_module import NER
from ner_config import DEFAULT_CONFIG

evidence_ner_model = NER()
opinion_ner_model = NER()

evidence_train_path = './data/ner_train_evidence.txt'
opinion_train_path = './data/ner_train_opinion.txt'
evidence_dev_path = './data/ner_dev_evidence.txt'
opinion_dev_path = './data/ner_dev_opinion.txt'
evidence_test_path = './data/ner_test_evidence.txt'
opinion_test_path = './data/ner_test_opinion.txt'
vec_path = os.path.join(DEFAULT_CONFIG['pretrained_path'], DEFAULT_CONFIG['pretrained_name'])

evidence_ner_model.train(evidence_train_path, dev_path=evidence_dev_path, save_path='./ner_saves',
                type="evidence")
opinion_ner_model.train(opinion_train_path, dev_path=opinion_dev_path, save_path='./ner_saves',
                type="opinion")
evidence_ner_model.load('./ner_saves', type="evidence")
opinion_ner_model.load('./ner_saves', type="opinion")
evidence_ner_model.test(evidence_test_path)
opinion_ner_model.test(opinion_test_path)
# predict = opinion_ner_model.predict(
#     "原代：真实性无异议，不能达到其证明目的，该份条款责任免除字迹太小，根据保险法以及保监会规定，其责任免除条款不符合规定，保险人未提供与第二被告的保险合同证明其以向第二被告尽到告知义务，而且系格式条款。责任的划分应在交强险外按4：6比例承担，被保险人系机动车，李来艳系非机动车，故而保险人应当承担6成。")
# print(predict)
# predict = opinion_ner_model.predict(
#     "被3代：证据一真实性无异议，原告为农村居民，应当按农村标准赔偿计算。证据二真实性无异议。证据三真实性无异议，但由于住院病历没有用药清单，是不完整的住院病历，不能证明其24张的发票是用于事故发生所致的伤害，对其自付的非医保用药部分我方不承担。证据四中的安庆市宜秀区大龙山镇永林居委会、安庆市宜秀区大龙山镇人民政府出具的证明有异议，土地征收应当有县级以上人民政府出具的土地征收文件，并附有征收人员名单及征收数量，故不能达到其证明目的。安庆市宜秀区大龙山镇中心小学证明有异议，在校学生应当有登记在册学籍，该份证据没有出具人员签名及学校负责人员签名，该份证据加盖的印章与落款不一致；证据五不具有客观真实性，与原告住院治疗的时间不稳合，同时也没有乘车人的姓名，不能达到其证明目的，对于交通费收条真实性、合法性均有异议，住宿费票据真实性、合法性、关联性均有异议，发票没有记载住宿人的姓名，同时该票据数额过高，不能证明因交通事故造成的损失；证据六是复印件，请求法庭核实。证据七有异议，是原告单方面所做的鉴定，鉴定明显过高，鉴定费发票真实性无异议，根据保险合同，保险公司不承担该费用。")
# print(predict)
# predict = evidence_ner_model.predict(
#     "被2代：一、营业执照、法定代表人身份证明各一份，证明被告身份。二、保单二份，证明第二被告在第三被告处投保的事实。")
# print(predict)
# predict = evidence_ner_model.predict(
#     "原代：第一组证据：李玉梅身份证、苏春身份证、李梓萌户口本、汪得青驾驶登记信息、皖HA5390号车辆基本信息表，证明原、被告的诉讼主体身份以及肇事车辆系安庆市鑫垚建材有限公司所有的基本信息。第二组证据：交通事故认定书，证明事故的发生经过以及责任的划分，李来艳和汪得青同责，李梓萌无责的事实。第三组证据：李梓萌入院记录一张，出院记录一张、疾病证明书一张、病程记录一张、安徽省立儿童医院病历一份、安庆市立医院门诊病历一份、MRI报告单一张、安庆市第一人民医院病历一份CT诊断报告单一张、X线检查报告单一张、医药费发票24张，证明原告因此次事故就医诊断的事实。")
# print(predict)
