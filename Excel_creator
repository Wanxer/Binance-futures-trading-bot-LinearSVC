import numpy as np
import pandas as pd
import talib,time,threading
from binance.helpers import *
from binance.client import Client
from binance.streams import BinanceSocketManager
import xlsxwriter
from contras import client_token


np_klines=np.array([])
open_ = np.array([])
high = np.array([])
low = np.array([])
pattern = []

client = client_token

workbook = xlsxwriter.Workbook('ethpredictdata.xlsx')
worksheet = workbook.add_worksheet("Eth_prices")
worksheet.write("A1","eth_prices")
klines = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1MINUTE, "8 days ago UTC")
for i in range(0,len(klines)-1):
    np_klines = np.append(np_klines, float(klines[i][1]))
    high = np.append(high, float(klines[i][2]))
    low = np.append(low, float(klines[i][3]))
    open_ = np.append(open_, float(klines[i][4]))
    if i > 0:    
        if np_klines[i-1] <  np_klines[i]:
            if ((np_klines[i]*100)/np_klines[i-1])-100 > 0.04:
              worksheet.write( "S" + str(i-1), 1)
              print(i)
        else: 
            worksheet.write( "S" + str(i-1), 0)
worksheet.write("S1", "up,down")

CCI = talib.CCI(high, low, np_klines, timeperiod=2)
aroon_up, aroon_down = talib.AROON(high, low, timeperiod = 2)
WILLR = talib.WILLR(high, low, np_klines, timeperiod=2)
fastk, fastd = talib.STOCHF(high, low, np_klines, fastk_period=2, fastd_period=3, fastd_matype=0)
ULTOSC = talib.ULTOSC(high, low, np_klines, timeperiod1=2, timeperiod2=4, timeperiod3=8)
upperband, middleband, lowerband = talib.BBANDS(np_klines, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
ULTOSC1 = talib.ULTOSC(high, low, np_klines, timeperiod1=3, timeperiod2=6, timeperiod3=12)
BOP = talib.BOP(open_, high, low, np_klines)
CCI1 = talib.CCI(high, low, np_klines, timeperiod=4)
aroon_up1, aroon_down1 = talib.AROON(high, low, timeperiod = 4)
WILLR1 = talib.WILLR(high, low, np_klines, timeperiod=4)
PLUS_DI = talib.PLUS_DI(high, low, np_klines, timeperiod=2)
PLUS_DI1 = talib.PLUS_DI(high, low, np_klines, timeperiod=5)
PLUS_DI2 = talib.PLUS_DI(high, low, np_klines, timeperiod=3)
MINUS_DI = talib.MINUS_DI(high, low, np_klines, timeperiod=2)
MINUS_DI1 = talib.MINUS_DI(high, low, np_klines, timeperiod=5)
MINUS_DI2 = talib.MINUS_DI(high, low, np_klines, timeperiod=3)



fastk = fastk.tolist()
worksheet.write("A1","fastk")
worksheet.write_column('A33', fastk[33:])

CCI = CCI.tolist()
worksheet.write("B1","CCI")
worksheet.write_column('B33', CCI[33:])

ULTOSC = ULTOSC.tolist()
worksheet.write("C1","ULTOSC")
worksheet.write_column('C33', ULTOSC[33:])

aroon_up = aroon_up.tolist()
worksheet.write("D1","aroon_up")
worksheet.write_column('D33', aroon_up[33:])

aroon_down = aroon_down.tolist()
worksheet.write("E1","aroon_down")
worksheet.write_column('E33', aroon_down[33:])

WILLR = WILLR.tolist()
worksheet.write("F1","WILLR")
worksheet.write_column('F33', WILLR[33:])

ULTOSC1 = ULTOSC1.tolist()
worksheet.write("G1","ULTOSC1")
worksheet.write_column('G33', ULTOSC1[33:])

aroon_up1 = aroon_up1.tolist()
worksheet.write("H1","aroon_up1")
worksheet.write_column('H33', aroon_up1[33:])

aroon_down1 = aroon_down1.tolist()
worksheet.write("I1","aroon_down1")
worksheet.write_column('I33', aroon_down1[33:])

WILLR1 = WILLR1.tolist()
worksheet.write("J1","WILLR1")
worksheet.write_column('J33', WILLR1[33:])

CCI1 = CCI1.tolist()
worksheet.write("K1","CCI1")
worksheet.write_column('K33', CCI1[33:])

BOP = BOP.tolist()
worksheet.write("L1","BOP")
worksheet.write_column('L33', BOP[33:])

PLUS_DI = PLUS_DI.tolist()
worksheet.write("M1","PLUS_DI")
worksheet.write_column('M33', PLUS_DI[33:])

PLUS_DI1 = PLUS_DI1.tolist()
worksheet.write("N1","PLUS_DI1")
worksheet.write_column('N33', PLUS_DI1[33:])

PLUS_DI2 = PLUS_DI2.tolist()
worksheet.write("O1","PLUS_DI2")
worksheet.write_column('O33', PLUS_DI2[33:])

MINUS_DI = MINUS_DI.tolist()
worksheet.write("P1","MINUS_DI")
worksheet.write_column('P33', MINUS_DI[33:])

MINUS_DI1 = MINUS_DI1.tolist()
worksheet.write("Q1","MINUS_DI1")
worksheet.write_column('Q33', MINUS_DI1[33:])

MINUS_DI2 = MINUS_DI2.tolist()
worksheet.write("R1","MINUS_DI2")
worksheet.write_column('R33', MINUS_DI2[33:])



workbook.close()
