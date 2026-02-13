import numpy as np 
import pandas as pd 
import talib,time,threading 
from binance.helpers import *
from binance.client import Client
from binance.streams import BinanceSocketManager 
import requests,json
import talib
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from time import sleep
from sklearn.svm import LinearSVC
from contras import client_token

def buyup():
    initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
    buyprice1=client.get_symbol_ticker(symbol="ETHUSDT")["price"]
    basicfee-=(basicfee*(0.02/100))*multiplicator
    ezfee-=(ezfee*(0.016/100))*multiplicator
    saveinfo(initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee)

def sellup():
    initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
    sell=client.get_symbol_ticker(symbol="ETHUSDT")["price"]
    winpercent=((float(sell)*100/float(buyprice1))-100)/100
    print("Benefits: " + str(basicfee*(winpercent)*multiplicator) + " " + str(ezfee*(winpercent)*multiplicator) + " " + str(minumfee*(winpercent)*multiplicator))
    basicfee+=basicfee*(winpercent)*multiplicator
    ezfee+=ezfee*(winpercent)*multiplicator
    minumfee+=minumfee*(winpercent)*multiplicator
    basicfee-=(basicfee*(0.02/100))*multiplicator
    ezfee-=(ezfee*(0.016/100))*multiplicator
    benefitbasic=basicfee-initialwallet
    benefitez=ezfee-initialwallet
    benefitminum=minumfee-initialwallet
    buyprice1=0
    saveinfo(initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee)


def buydown():
    initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
    sellprice1=client.get_symbol_ticker(symbol="ETHUSDT")["price"]
    basicfee-=(basicfee*(0.02/100))*multiplicator
    ezfee-=(ezfee*(0.016/100))*multiplicator
    saveinfo(initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee)


def selldown():
    initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
    sell=client.get_symbol_ticker(symbol="ETHUSDT")["price"]
    winpercent=(((float(sell)*100/float(sellprice1))-100)/100)*-1
    print("Benefits: " + str(basicfee*(winpercent)*multiplicator) + " " + str(ezfee*(winpercent)*multiplicator) + " " + str(minumfee*(winpercent)*multiplicator))
    basicfee+=basicfee*(winpercent)*multiplicator
    ezfee+=ezfee*(winpercent)*multiplicator
    minumfee+=minumfee*(winpercent)*multiplicator
    basicfee-=(basicfee*(0.02/100))*multiplicator
    ezfee-=(ezfee*(0.016/100))*multiplicator
    benefitbasic=basicfee-initialwallet
    benefitez=ezfee-initialwallet
    benefitminum=minumfee-initialwallet
    sellprice1=0
    saveinfo(initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee)


def loadinfo():
  load=open("info.txt", "r")
  info=load.read()
  load.close()

  initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = info.split()
  initialwallet=float(initialwallet)
  multiplicator=int(multiplicator)
  buyprice1=float(buyprice1)
  sellprice1=float(sellprice1)
  benefitbasic=float(benefitbasic)
  benefitez=float(benefitez)
  benefitminum=float(benefitminum)
  basicfee=float(basicfee)
  ezfee=float(ezfee)
  minumfee=float(minumfee)
  if str(ihave1) == "True":
    ihave1=True
  else:
    ihave1=False
  return initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee
initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
def saveinfo(initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee):
  archiu=open("info.txt", "w")
  archiu.write(str(initialwallet) + " " + str(multiplicator) + " " + str(ihave1) + " " + str(buyprice1) + " " + str(sellprice1) + " " + str(benefitbasic) + " " + str(benefitez) + " " + str(benefitminum) + " " + str(basicfee) + " " + str(ezfee) + " " + str(minumfee))
  archiu.close

client = client_token

############################################################################################################################################

normal_data=[]
dat=[]
this_data=[]
data = pd.read_excel("ethpredictdata.xlsx")
data = data.values.tolist()[80:-2]
for i in range(0, len(data)):
    normal_data.append(data[i][0])
    normal_data.append(((data[i][1])+500)/10)
    normal_data.append((data[i][2]))
    normal_data.append((data[i][3]))
    normal_data.append((data[i][4]))
    normal_data.append((data[i][5])+100)
    normal_data.append(((data[i][6])+1)*50)
    normal_data.append(data[i][7]*2.5)
    normal_data.append(data[i][8]*2.5)
    normal_data.append((data[i][9]*100))
    dat.append(normal_data)
    normal_data=[]
featuresl=[]
labelsl=[]
for line in dat:
    featuresl.append(line[:9])
    labelsl.append(line[-1])
featuresl=np.array(featuresl)
labelsl=np.array(labelsl)
print(featuresl[-1])
print(labelsl[-1])
X_train, X_test, y_train, y_test = train_test_split(featuresl, labelsl, test_size=0.1, random_state=106)

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy up:",metrics.accuracy_score(y_test, y_pred))

############################################################################################################################################

normal_data=[]
dat=[]
this_data=[]
data = pd.read_excel("ethpredictdata.xlsx")
data = data.values.tolist()[80:-2]
for i in range(0, len(data)):
    normal_data.append(data[i][0])
    normal_data.append(((data[i][1])+500)/10)
    normal_data.append((data[i][2]))
    normal_data.append((data[i][3]))
    normal_data.append((data[i][4]))
    normal_data.append((data[i][5])+100)
    normal_data.append(((data[i][6])+1)*50)
    normal_data.append(data[i][7]*2.5)
    normal_data.append(data[i][8]*2.5)
    normal_data.append((data[i][10]*100))
    dat.append(normal_data)
    normal_data=[]
featuresl=[]
labelsl=[]
for line in dat:
    featuresl.append(line[:9])
    labelsl.append(line[-1])
featuresl=np.array(featuresl)
labelsl=np.array(labelsl)

X_train, X_test, y_train, y_test = train_test_split(featuresl, labelsl, test_size=0.1, random_state=106)

clf2 = LinearSVC(random_state=0, tol=1e-5)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
print("Accuracy up:",metrics.accuracy_score(y_test, y_pred))

############################################################################################################################################

status1="a"
lastmin=-1
periods=0
sellprice1 = 0


time1=datetime.now()
start=open("start.txt", "w")
start.write(str(time1))
start.close
while True:
    if True:
      time.sleep(8)
      continua = False
      np_klines=np.array([])
      open_ = np.array([])
      high = np.array([])
      low = np.array([])
      if True:
          klines = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE)
          for i in range(0,len(client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE))):
              np_klines = np.append(np_klines, float(klines[i][1]))
              open_ = np.append(open_, float(klines[i][4]))
              high = np.append(high, float(klines[i][2]))
              low = np.append(low, float(klines[i][3]))
          print(np_klines[-1])
          print(open_[-1])
          CCI = talib.CCI(high, low, np_klines, timeperiod=2)
          aroon_up, aroon_down = talib.AROON(high, low, timeperiod = 2)
          WILLR = talib.WILLR(high, low, np_klines, timeperiod=2)
          fastk, fastd = talib.STOCHF(high, low, np_klines, fastk_period=2, fastd_period=3, fastd_matype=0)
          ULTOSC = talib.ULTOSC(high, low, np_klines, timeperiod1=2, timeperiod2=4, timeperiod3=8)
          upperband, middleband, lowerband = talib.BBANDS(np_klines, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
          BOP = talib.BOP(open_, high, low, np_klines)
          PLUS_DI = talib.PLUS_DI(high, low, np_klines, timeperiod=2)
          MINUS_DI = talib.MINUS_DI(high, low, np_klines, timeperiod=2)


          
          
          actual_fastk = fastk[-1]
          actual_CCI = CCI[-1]
          actual_ULTOSC = ULTOSC[-1]
          actual_aroondown = aroon_down[-1]
          actual_aroonup = aroon_up[-1]
          actual_WILLR = WILLR[-1]
          actual_BOP = BOP[-1]
        
          
          
          scalede=[actual_fastk, (actual_CCI+500)/10, actual_ULTOSC, actual_aroonup, actual_aroondown, actual_WILLR+100,
                      (actual_BOP+1)*50, PLUS_DI[-1] * 2.5, MINUS_DI[-1] *2.5 ]
          indicators=open("indicators.txt", "w")
          indicators.write(str(scalede))
          indicators.close()
          scalede = np.array(scalede).reshape(1, -1)

          up = clf.predict(scalede)
          if up != 100:
              down = clf2.predict(scalede)
          else:
              down = 0

          if up == 100:
              if buyprice1 == 0:
                  if sellprice1 != 0:
                      selldown()
                      initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
                  buyup()
                  initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
          elif up == 0:
              if buyprice1 != 0:
                 sellup()
                 initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
          if down == 100:
              if sellprice1 == 0:
                 if buyprice1 != 0:
                     sellup()
                     initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
                 buydown()
                 initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
          elif down == 0:
              if sellprice1 != 0:
                 selldown()
                 initialwallet, multiplicator, ihave1, buyprice1, sellprice1, benefitbasic, benefitez, benefitminum, basicfee, ezfee, minumfee = loadinfo()
                      
              

          print("Benefits: ", benefitbasic, benefitez, benefitminum)
          print("Period finished at ", datetime.now(), "\n \n")
        
    lastmin=datetime.now().minute
