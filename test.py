from text2vec import SentenceModel
import requests
import os

os.environ['http_proxy'] = 'http://127.0.0.1:10810'
os.environ['https_proxy'] = os.environ['http_proxy']

requests.get("https://www.google.com.hk", timeout=5)
requests.get("https://www.baidu.com", timeout=5)
m = SentenceModel()
m.encode("如何更换花呗绑定银行卡")
