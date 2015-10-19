import urllib2
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from urllib import urlopen
import json

try:
    # py3
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
except ImportError:
    # py2
    from urllib2 import Request, urlopen
    from urllib import urlencode

logreturns = []
variancecalcs = [] 
annualvolprime = float()
annualvolprime1 = []
pi = 3.14159265358979 

daysToMaturityPrime = float()
rate = .0015
q = float()
spot = float()

puts_impliedvols = []
calls_impliedvols = []

months = []

adjclose = [] 
date = []
opent = []
high = []
low = []
close = []
volume = []
expirys = []

urlToVisit = "http://ichart.finance.yahoo.com/table.csv?s="
# Y, M, D
start = datetime.date(2014, 9, 10)
end = datetime.date.today()
        
        
def googleQuote(ticker):
    chain = []
    url = 'http://www.google.com/finance/option_chain?q=%s&output=json'% ticker
    content = urlopen(url).read()
    a = fix_json(content)
    #print a
    opts = eval(a)
    exp = opts['expirations']

    for expiry in exp:
        y = expiry['y']
        m = expiry['m']
        d = expiry['d']
        url = 'http://www.google.com/finance/option_chain?q=%s&output=json&expy=%s&expm=%s&expd=%s'%(ticker,y,m,d)
        lines = fix_json(urllib2.urlopen(url).read())
        chain.append(lines)

    #prints out option chain data for each expiry date
    #i = 1
    #for lines in chain:
    #    quote = json.loads(lines)
    #    print i, ' : ', quote, '\n'
    #    i += 1

    return chain

def fix_json(k):
    q=['cid','cp','s','cs','vol','expiry','underlying_id','underlying_price',
     'p','c','oi','e','b','strike','a','name','puts','calls','expirations',
     'y','m','d']
 
    for i in q:
        try:    
            k=k.replace('{%s:'%i,'{"%s":'%i)
            k=k.replace(',%s:'%i,',"%s":'%i)
        except: pass
 
    return k

def makeUrl(stock, start, end):
    a = start
    b = end
    dateUrl = '%s&a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&g=d&ignore=.csv'% (stock, a.month-1, a.day, a.year, b.month-1, b.day, b.year)
    return urlToVisit+dateUrl

def pullData(stock,cut):
    get_quote(stock)
    data = googleQuote(stock)
    try: 
        print "Currently pulling", stock
        stockUrl = makeUrl(stock, start, end)
        stockFile = []
        try:
            sourceCode = urllib2.urlopen(stockUrl).read()
            splitSource = sourceCode.split('\n')
            
            for eachLine in splitSource:
                splitLine = eachLine.split(',')
                if len(splitLine) == 7:
                    if 'values' not in eachLine:
                        stockFile.append(eachLine)
        except Exception, e:
            print str(e), 'failed to organize pulled data'
    except Exception, e:
        print str(e), 'failed to pull stock historical data'
        
    try:
        for line in stockFile:
            s = line.split(',')

            if s[0] == 'Date' or s[1] == 'Open' or s[2] == 'High' or s[3] == 'Low' or s[4] == 'Close' or s[5] == 'Volume' or s[6] == 'Adj Close':
                pass
            else:
                date_obj = datetime.datetime.strptime(s[0], '%Y-%m-%d').date()
                date.append(date_obj)
                opent.append(s[1])
                high.append(s[2])
                low.append(s[3])
                close.append(s[4])
                low.append(s[5])
                adjclose.append(s[6])
    except Exception, e:
        print str(e), 'error'


    #graph figure initialization
    fig = plt.figure()
    fig.suptitle('Volatility Smile, Term and Surface', fontsize=18)
    ########################################

    #Stock graph initialization
    ########################################
    plt.subplot(224)
    plt.ylabel("AdjClose")
    plt.xlabel('DATE')
    plt.title('STOCK PRICE - DATE', color='#000000')
    plt.plot(date, adjclose, 'bo')
    plt.plot(date, adjclose, 'k--')

    #vol smile initialization
    ########################################
    plt.subplot(221)
    createExpiryDate = datetime.datetime.today() + datetime.timedelta(days=daysToMaturityPrime)
    user_Day = createExpiryDate.day
    user_Month = createExpiryDate.month
    user_Year = createExpiryDate.year
    # print 'YEAR', user_Year, 'MONTH', user_Month, 'DAY', user_Day
    # print "NEW EXPIRY DATE: ", createExpiryDate
    url = 'http://www.google.com/finance/option_chain?q=%s&output=json&expy=%s&expm=%s&expd=%s'%(stock,user_Year,user_Month,user_Day)
    lines = fix_json(urllib2.urlopen(url).read())
    quote = json.loads(lines)
    call_strike = []
    put_strike = []
    call_mid = []
    put_mid = []
    print quote
    puts = quote['puts']
    calls = quote['calls']
    for c in calls:
        if c['b'] == '-' or c['a'] == '-':
            pass
        else:
            call_strike.append(float(c['strike']))
            mid = (float(c['b']) + float(c['a'])) / 2
            call_mid.append(mid)
    for p in puts:
        if p['b'] == '-' or p['a'] == '-':
            pass
        else:
            put_strike.append(float(p['strike']))
            mid = (float(p['b']) + float(p['a'])) / 2
            put_mid.append(mid)
    impliedVolWithStikes(adjclose, call_strike, put_strike, call_mid, put_mid, createExpiryDate.date(), 'volsmile', cut)
    del puts_impliedvols[:]
    del calls_impliedvols[:]
    ########################################

    #3D curve initiliazation
    ########################################
    #replace ax initialization with this if only graphing 3D curve
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(223, projection='3d')

    main3DImpVol=[]
    main3DStrike=[]
    main3DDates=[]
    termStructDates=[]
    termStructImpVols=[]
    ########################################

    #i = 1
    for line in data:
        curDate3Dz=[]
        strike_call = []
        strike_put = []
        mid_call = []
        mid_put = []
        quote = json.loads(line)
        puts = quote['puts']
        calls = quote['calls']
        expiry_date = datetime.datetime.strptime(calls[1]['expiry'], '%b %d, %Y')
    
        expirys.append(expiry_date)
        d = expiry_date - datetime.datetime.today()
        global daysToMaturityPrime
        daysToMaturityPrime = d.total_seconds() / (3600 * 24)
        for c in calls:
            if c['b'] == '-' or c['a'] == '-':
                pass
            else:
                strike_call.append(float(c['strike']))
                mid = (float(c['b']) + float(c['a'])) / 2
                mid_call.append(mid)
        for p in puts:
            if p['b'] == '-' or p['a'] == '-':
                pass
            else:
                strike_put.append(float(p['strike']))
                mid = (float(p['b']) + float(p['a'])) / 2
                mid_put.append(mid)

        impliedVolWithStikes3D(adjclose, strike_call, strike_put, mid_call, mid_put, 'volsmile', cut)

        for i in range(0,len(calls_impliedvols)): 
           curDate3Dz.append(daysToMaturityPrime)

        print "-------------------------------"
        print "len(strike_call): ",             len(strike_call)
        print "len(calls_impliedvols): ",       len(calls_impliedvols)
        print "len(curDate3Dz): ",              len(curDate3Dz)
        print "-------------------------------"
        #ax.plot_trisurf(strike_call, calls_impliedvols, curDate3Dz, cmap=cm.jet, linewidth=0.2)
        
        main3DImpVol.extend(calls_impliedvols)
        main3DStrike.extend(strike_call)
        main3DDates.extend(curDate3Dz)

        del curDate3Dz[:]
        curDate3Dz=[]
        for i in range(0,len(puts_impliedvols)): 
            curDate3Dz.append(daysToMaturityPrime)
        
        print "-------------------------------"
        print "len(strike_put): ",              len(strike_put)
        print "len(puts_impliedvols): ",        len(puts_impliedvols)
        print "len(curDate3Dz): ",              len(curDate3Dz)
        print "-------------------------------"
        #ax.plot_trisurf(strike_put, puts_impliedvols, curDate3Dz, cmap=cm.jet, linewidth=0.2)
        
        main3DImpVol.extend(puts_impliedvols)
        main3DStrike.extend(strike_put)
        main3DDates.extend(curDate3Dz)

        #Filling arrays for term structure volatility
        termStructDates.append(daysToMaturityPrime)
        totalAverage=(sum(puts_impliedvols)/float(len(puts_impliedvols)))+(sum(calls_impliedvols)/float(len(calls_impliedvols)))
        totalAverage=totalAverage/2
        termStructImpVols.append(totalAverage)

        del puts_impliedvols[:]
        del calls_impliedvols[:]
        # print 'RAN', i, 'TIMES\n'
        # i += 1

    ax.plot_trisurf(main3DStrike, main3DDates, main3DImpVol, cmap=cm.jet, linewidth=0.2)
    ax.set_xlabel('STRIKES')
    ax.set_ylabel('# DAYS to MAT.')
    ax.set_zlabel('IMP. VOL.')

    ########################################
    #Term Structure Volatility initiliazation
    ########################################
    plt.subplot(222)
    plt.ylabel("Implied Option Volatility")
    plt.xlabel('# DAYS to MAT.')
    plt.title('TERM STRUCTURE VOL.', color='#000000')
    termPoints,=plt.plot(termStructDates, termStructImpVols, 'ro')
    plt.plot(termStructDates, termStructImpVols, 'k--')
    # legend = plt.legend(termPoints,"Avrg. ImpVol",loc='upper right', shadow=True, numpoints=1)
    # frame = legend.get_frame()
    # frame.set_facecolor('0.90')
    # for label in legend.get_texts():
    #     label.set_fontsize('large')
    # for label in legend.get_lines():
    #     label.set_linewidth(1.5) 

    plt.show()
        
def _request(symbol, stat):
    url = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (symbol, stat)
    req = Request(url)
    resp = urlopen(req)
    content = resp.read().decode().strip()
    return content
    
def get_quote(symbol):
    ids = 'yl1'
    values = _request(symbol, ids).split(',')
    global q
    global spot

    q = float(values[0]) / 100 
    spot = float(values[1])
    
def impliedVolWithStikes(adjclose, strikes, strikesput, callMid, putMid, expiry_date, type, cut):
    logreturn(adjclose)
    variancecalc(logreturns)
    annualvol(stdev(varianceaverage(variancecalcs)))
    print "Spot:                                    ", spot
    print "Historical annual volalitility for Call: ", annualvolprime    

    i=0;
    j=0;
    maxCallImpVol=0;
    if(type == 'volsmile'):

        while i < len(strikes):
            #call option premium calculation
            option = OptionPrice(spot, strikes[i], daysToMaturityPrime, annualvolprime, rate, q, "c")
            # print "Theoretical of Call: $", option
            # print "Market Price of Call: $", callMid[i]
            #print "Input Check: ", "spot: ", spot, "strike: ", strikes[i], "rate: ", rate, "q (div)", q, "days to maturityPrime: ", daysToMaturityPrime
            
            #call option implied volatility calculation
            impliedvol = calls_annualvolimplied(option, callMid[i], strikes[i], "c",)

            #calculating maximum impliedvol for calls, <=1. x/2 of this used for spot line length
            if impliedvol<=1:
                if impliedvol>maxCallImpVol:
                    maxCallImpVol=impliedvol
            # print "Strike:                                  ", strikes[i]
            # print "Historical annual volalitility for Call: ", annualvolprime
            # print "Implied annual volalitity for Call:      ", impliedvol
            # print ""
            i += 1

        while j < len(strikesput):
            #put option premium calculation
            option2 = OptionPrice(spot, strikesput[j], daysToMaturityPrime, annualvolprime, rate, q, "p")
            # print "Theoretical of Put: $", option2
            # print "Market Price of Put: $", putMid[j]
            #print "Input Check: ", "spot: ", spot, "strike: ", strikesput[j], "rate: ", rate, "q (div)", q, "days to maturityPrime: ", daysToMaturityPrime
            
            #call option implied volatility calculation
            impliedvol2 = puts_annualvolimplied(option2, putMid[j], strikesput[j], "p")
            # print "Strike:                                 ", strikesput[j]
            # print "Historical annual volalitility for Put: ", annualvolprime
            # print "Implied annual volalitity for Put:      ", impliedvol2
            # print ""
            j += 1

    if(type == 'volsmile'):
        #plt.figure()
        
        if cut == 'Y':
            putsLength=len(puts_impliedvols)
            callsLength=len(calls_impliedvols)
            diff=callsLength-putsLength

            #puts on left side count
            minStrikePut=spot
            counterPutsLeft=0
            for i in strikesput:
                if i<spot:
                    minStrikePut=i
                    counterPutsLeft+=1
                    
            #calls on left side count
            counterCallsLeft=0
            for i in strikes:
                if i<spot:
                    counterCallsLeft+=1

            print "putsLength: ",       putsLength
            print "callsLength: ",      callsLength
            print "counterPutsLeft: ",  counterPutsLeft

            #getting rid of first outlier for OTM puts
            if len(strikesput)>0:
                strikesput.pop(0)
                puts_impliedvols.pop(0)

            #cutting.
            #for example: 
            #16 puts below spot, so want graph 16 calls below spot
            #currently have <counterCallsLeft> calls below spot 
            #16=counterCallsLeft-x (x is how many pops)
            x=counterCallsLeft-counterPutsLeft
            diff=callsLength-putsLength
            if diff>0:
                for i in range(0,x):
                    strikes.pop(0)
                    calls_impliedvols.pop(0)

        print "maxCallImpVol: ", maxCallImpVol
        plt.axvline(x=spot, ymin=0, ymax=(maxCallImpVol/2), linewidth=2, linestyle='dashed', color='k')
        
        if cut == 'Y':
            plt.text(spot,(maxCallImpVol/2),('ATM\n($%s)'%spot),rotation=0)
            plt.text(spot+20,(maxCallImpVol/2)+0.1,('----------->\nOTM Calls\nITM Puts'),rotation=0)
            plt.text(spot-30,(maxCallImpVol/2)+0.1,('<-----------\nITM Calls\nOTM Puts'),rotation=0)
            plt.text(spot, (maxCallImpVol/1.25), ('Expiry: %s'%(str(expiry_date))),rotation=0)

        print "strikesput length:",         len(strikesput)
        print "puts_impliedvols length:",   len(puts_impliedvols)
        print "strikescalls length:",       len(strikes)
        print "calls_impliedvols length:",  len(calls_impliedvols)
        putsPoints,=plt.plot(strikesput, puts_impliedvols, 'bo')
        callsPoints,=plt.plot(strikes, calls_impliedvols, 'ro')
        plt.plot(strikesput, puts_impliedvols, 'k--', strikes, calls_impliedvols, 'k--')

        legend = plt.legend([putsPoints,callsPoints],["Puts", "Calls"],loc='upper right', shadow=True, numpoints=1)
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

        for label in legend.get_texts():
            label.set_fontsize('large')
        for label in legend.get_lines():
            label.set_linewidth(1.5) 

        #axis titles
        plt.ylabel("Implied Option Volatility")
        plt.xlabel("Strike")
        plt.title('STRIKE - VOLATILITY SMILE', color='#000000')

        #display
        #plt.show()


    i = 0;
    if(type == 'c'):
        while i < len(strikes):
            option = OptionPrice(spot, strikes[i], daysToMaturityPrime, annualvolprime, rate, q, "c")
            print "Theoretical of Call: $", option
            print "Market Price of Call: $", callMid[i]
            print "Input Check: ", "spot: ", spot, "strike: ", strikes[i], "rate: ", rate, "q (div)", q, "days to maturityPrime: ", daysToMaturityPrime
            
            impliedvol = calls_annualvolimplied(option, callMid[i], strikes[i], "c",)
        
            print "Strike:                                  ", strikes[i]
            print "Historical annual volalitility for Call: ", annualvolprime
            print "Implied annual volalitity for Call:      ", impliedvol
            print ""
            i += 1

    if(type == 'c'):
        plt.axvline(x=spot, ymin=0, ymax = 0.7, linewidth=2, color='k')

        plt.plot(strikes, calls_impliedvols, 'ro')
        #plt.plot(strikes, calls_impliedvols, 'k')
        plt.ylabel("Implied Call Option Volatility")
        plt.xlabel("Strike")
        plt.title('Verticle IVol Skew', color='#000000')
        plt.show()

    i = 0;
    if(type == 'p'):
        print "GIVE ME SOME PUT SPACE"
        while i < len(strikes):
            option = OptionPrice(spot, strikes[i], daysToMaturityPrime, annualvolprime, rate, q, "p")
            print "Theoretical of Put: $", option
            print "Market Price of Put: $", putMid[i]
            print "Input Check: ", "spot: ", spot, "strike: ", strikes[i], "rate: ", rate, "q (div)", q, "days to maturityPrime: ", daysToMaturityPrime
            
            impliedvol = puts_annualvolimplied(option, putMid[i], strikes[i], "p")

            print "Strike:                                  ", strikes[i]
            print "Historical annual volalitility for Put: ", annualvolprime
            print "Implied annual volalitity for Put:      ", impliedvol
            print ""
            i += 1
    if(type == 'p'):
        plt.figure()

        temp1=strikes
        temp2=calls_impliedvols



        plt.plot(strikes, puts_impliedvols, 'ro')
        #plt.plot(strikes, calls_impliedvols, 'k')
        plt.ylabel("Implied Put Option Volatility")
        plt.axvline(x=spot, ymin=0, ymax = 0.7, linewidth=2, color='k')
        plt.xlabel("Strike")
        plt.title('Verticle IVol Skew', color='#000000')
        plt.show()
    return


def impliedVolWithStikes3D(adjclose, strikes, strikesput, callMid, putMid, type, cut):
    logreturn(adjclose)
    variancecalc(logreturns)
    annualvol(stdev(varianceaverage(variancecalcs)))
    print "Spot:                                    ", spot
    print "Historical annual volalitility for Call: ", annualvolprime    

    i = 0;
    j=0;
    maxCallImpVol=0;
    if(type == 'volsmile'):

        while i < len(strikes):
            #call option premium calculation
            option = OptionPrice(spot, strikes[i], daysToMaturityPrime, annualvolprime, rate, q, "c")
            # print "Theoretical of Call: $", option
            # print "Market Price of Call: $", callMid[i]
            #print "Input Check: ", "spot: ", spot, "strike: ", strikes[i], "rate: ", rate, "q (div)", q, "days to maturityPrime: ", daysToMaturityPrime
            
            #call option implied volatility calculation
            impliedvol = calls_annualvolimplied(option, callMid[i], strikes[i], "c",)

            #calculating maximum impliedvol for calls, <=1. x/2 of this used for spot line length
            if impliedvol<=1:
                if impliedvol>maxCallImpVol:
                    maxCallImpVol=impliedvol
            # print "Strike:                                  ", strikes[i]
            # print "Historical annual volalitility for Call: ", annualvolprime
            # print "Implied annual volalitity for Call:      ", impliedvol
            # print ""
            i += 1

        while j < len(strikesput):
            #put option premium calculation
            option2 = OptionPrice(spot, strikesput[j], daysToMaturityPrime, annualvolprime, rate, q, "p")
            # print "Theoretical of Put: $", option2
            # print "Market Price of Put: $", putMid[j]
            #print "Input Check: ", "spot: ", spot, "strike: ", strikesput[j], "rate: ", rate, "q (div)", q, "days to maturityPrime: ", daysToMaturityPrime
            
            #call option implied volatility calculation
            impliedvol2 = puts_annualvolimplied(option2, putMid[j], strikesput[j], "p")
            # print "Strike:                                 ", strikesput[j]
            # print "Historical annual volalitility for Put: ", annualvolprime
            # print "Implied annual volalitity for Put:      ", impliedvol2
            # print ""
            j += 1

    if(type == 'volsmile'):
        
        if cut == 'Y':
            putsLength=len(puts_impliedvols)
            callsLength=len(calls_impliedvols)
            diff=callsLength-putsLength

            #puts on left side count
            minStrikePut=spot
            counterPutsLeft=0
            for i in strikesput:
                if i<spot:
                    minStrikePut=i
                    counterPutsLeft+=1
                    
            #calls on left side count
            counterCallsLeft=0
            for i in strikes:
                if i<spot:
                    counterCallsLeft+=1

            print "putsLength: ",       putsLength
            print "callsLength: ",      callsLength
            print "counterPutsLeft: ",  counterPutsLeft

            #getting rid of first outlier for OTM puts
            if len(strikesput)>0:
                strikesput.pop(0)
                puts_impliedvols.pop(0)

            #cutting.
            #for example: 
            #16 puts below spot, so want graph 16 calls below spot
            #currently have <counterCallsLeft> calls below spot 
            #16=counterCallsLeft-x (x is how many pops)
            x=counterCallsLeft-counterPutsLeft
            diff=callsLength-putsLength
            if diff>0:
                for i in range(0,x):
                    strikes.pop(0)
                    calls_impliedvols.pop(0)

        print "strikesput length:",       len(strikesput)
        print "puts_impliedvols length:", len(puts_impliedvols)
        print "strikescalls length:",       len(strikes)
        print "calls_impliedvols length:", len(calls_impliedvols)
    return



            
# def timeToMaturity(year, month, day):
#     maturityDate = datetime.date(year, month, day)
#     difference = maturityDate - end
#     global daysToMaturityPrime
#     daysToMaturityPrime = (difference.total_seconds() / (3600 * 24))
#     return daysToMaturityPrime
        
def OptionPrice(spot, strike, NbExp, vol, rate, q, optionType):
    # v = float()
    d1 = float()
    d2 = float()
    Nd1 = float()
    Nd2 = float()
    T = float()
        
    if NbExp < 0:
        return 0
    T = NbExp / 365
    if NbExp == 0:
        if optionType == "c":
            print "TESTTESTTESTTEST", long((math.max(spot - strike, 0)))
            return float((math.max(spot - strike, 0)))
        else:
            print "TESTTESTTESTTEST", long((math.max(spot - strike, 0)))
            return float((math.max(strike - spot, 0)))
    
    d1 = ((math.log(spot / strike)) + (rate - q + (vol * vol) / 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    Nd1 = cdnf(d1)
    Nd2 = cdnf(d2)
    if optionType == "c":
        # call option
        return float((spot * math.exp(-q * T) * Nd1 - strike * math.exp(-rate * T) * Nd2))
    else:
        # put option
        return float((-spot * math.exp(-q * T) * (1 - Nd1) + strike * math.exp(-rate * T) * (1 - Nd2)))
  
def greeks(spot, strike, NbExp, vol, rate, q, optionType):
    dS = float()
    dv = float()
    dr = float()
    dt = float()
    delta = float()
    gamma = float()
    vega = float()
    theta = float()
    rho = float()
    dS = 0.01
    # 0.01 point move in spot
    dv = 0.0001
    # 0.01% move in vol
    dt = 1
    # 1 day
    dr = 0.0001
    # 1bps move
    if NbExp < 0:
        # print "TESTTESTTESTTEST";
        return 0
    #x = float((cls.OptionPrice(spot + dS, strike, NbExp, vol, rate, q, optionType)))
    #x2 = float(cls.OptionPrice(spot - dS, strike, NbExp, vol, rate, q, optionType))
    # print x;
    # print x2;
    # print dS;
    # print (x-x2)/(2*dS);
    delta = float(((OptionPrice(spot + dS, strike, NbExp, vol, rate, q, optionType) - OptionPrice(spot - dS, strike, NbExp, vol, rate, q, optionType)) / (2 * dS)))
    gamma = float(((OptionPrice(spot + dS, strike, NbExp, vol, rate, q, optionType) - 2 * OptionPrice(spot, strike, NbExp, vol, rate, q, optionType) + OptionPrice(spot - dS, strike, NbExp, vol, rate, q, optionType)) / (dS * dS)))
    vega = float(((OptionPrice(spot, strike, NbExp, vol + dv, rate, q, optionType) - OptionPrice(spot, strike, NbExp, vol - dv, rate, q, optionType)) / (2 * dv) / 100))
    rho = float(((OptionPrice(spot, strike, NbExp, vol, rate + dr, q, optionType) - OptionPrice(spot, strike, NbExp, vol, rate - dr, q, optionType)) / (2 * dr) / 1000))
    if NbExp == 0:
        # print "TESTTESTTESTTEST";
        theta = 0
    else:
        theta = float(((OptionPrice(spot, strike, NbExp - dt, vol, rate, q, optionType) - OptionPrice(spot, strike, NbExp + dt, vol, rate, q, optionType)) / (2 * dt)))
    print "delta: ", delta
    print "gamma: ", gamma
    print "vega:  ", vega
    print "rho>:  ", rho
    print "theta: ", theta
    return 0
                 
def cdnf(x):
    neg = 1 if (x < 0) else 0
    if neg == 1:
        x *= -1
    k = (1 / (1 + 0.2316419 * x))
    y = ((((1.330274429 * k - 1.821255978) * k + 1.781477937) * k - 0.356563782) * k + 0.319381530) * k
    y = 1.0 - 0.398942280401 * math.exp(-0.5 * x * x) * y
    return (1 - neg) * y + neg * (1 - y)
        
def logreturn(adjclose):
    i = 0
    while i < len(adjclose) - 1:
        x = math.log(float(adjclose[i])/float(adjclose[i+1]))
        global logreturns
        logreturns.append(x)
        i += 1
        
def averagelog(logreturns):
    sum = 0
    counter = 0
    i = 0
    while i < len(logreturns) - 1:
        y = logreturns[i]
        sum += y
        counter += 1
        i += 1
    x = sum / counter
    return x
    
def variancecalc(logreturns):
    i = 0
    avg = averagelog(logreturns)
    while i < len(logreturns) - 1:
        y = logreturns[i]
        yprime = y - avg
        global variancecalcs
        variancecalcs.append(math.pow(yprime, 2))
        i += 1

def varianceaverage(variancecalc):
    sum = 0
    counter = 0
    i = 0
    while i < len(variancecalc) - 1:
        y = variancecalc[i]
        sum += y
        counter += 1
        i += 1
    #counter should be n-1 to get an unbiased estimate
    x = sum / (len(variancecalc) - 1)
    #print "**********************************"
    #print "VarAvrg:", x
    return x


def stdev(varianceAvrg):
    return math.sqrt(varianceAvrg)


def annualvol(stdev):
    annualvol = math.sqrt(252) * stdev
    global annualvolprime 
    annualvolprime = annualvol
    #this should add historial vols by month in sequence
    annualvolprime1.append(annualvol)
    return annualvol
    
def annualvol2(stdev):
    return math.sqrt(daysToMaturityPrime) * stdev
    
def calls_annualvolimplied(modelOption, realOption, strike, optionType):
        volimplied = stdev(varianceaverage(variancecalcs))
        calls_annualvolimplied = 0
        real = realOption
        model = modelOption
        if real == model:
            calls_annualvolimplied = annualvol2(volimplied)
        elif real > model:
            while (real > model):
                volimplied += 0.00001
                model = OptionPrice(spot, strike, daysToMaturityPrime, annualvol2(volimplied), rate, q, optionType)
                calls_annualvolimplied = annualvol2(volimplied)
                #print 'real > model', calls_annualvolimplied
                if calls_annualvolimplied < 0:
                    #print 'ERROR: OUT OF THE MONEY'
                    break
        else:
            while (real < model):
                volimplied -= 0.00001
                model = OptionPrice(spot, strike, daysToMaturityPrime, annualvol2(volimplied), rate, q, optionType)
                calls_annualvolimplied = annualvol2(volimplied)
                #print 'real < model', calls_annualvolimplied
                if calls_annualvolimplied < 0:
                    #print 'ERROR: OUT OF THE MONEY'
                    break
        
        calls_impliedvols.append(calls_annualvolimplied)
        return calls_annualvolimplied

def puts_annualvolimplied(modelOption, realOption, strike, optionType):
        volimplied = stdev(varianceaverage(variancecalcs))
        puts_annualvolimplied = 0
        real = realOption
        model = modelOption
        if real == model:
            puts_annualvolimplied = annualvol2(volimplied)
        elif real > model:
            while (real > model):
                volimplied += 0.00001
                model = OptionPrice(spot, strike, daysToMaturityPrime, annualvol2(volimplied), rate, q, optionType)
                puts_annualvolimplied = annualvol2(volimplied)
                #print 'real > model', calls_annualvolimplied
                if puts_annualvolimplied < 0:
                    # print 'ERROR: OUT OF THE MONEY'
                    break
        else:
            while (real < model):
                volimplied -= 0.00001
                model = OptionPrice(spot, strike, daysToMaturityPrime, annualvol2(volimplied), rate, q, optionType)
                puts_annualvolimplied = annualvol2(volimplied)
                #print 'real < model', calls_annualvolimplied
                if puts_annualvolimplied < 0:
                    # print 'ERROR: OUT OF THE MONEY'
                    break
             
        puts_impliedvols.append(puts_annualvolimplied)
        return puts_annualvolimplied

while True:
    stock = raw_input('Stock to pull: ')
    # spot = float(raw_input('Spot: '))
    #strike = float(raw_input('Strike: '))
    #realOptionPrice = float(raw_input('Real Option Price: '))
    daysToMaturityPrime = float(raw_input('Days to Maturity: '))
    #rate = float(raw_input('Risk-free Rate: '))
    # q = float(raw_input('Div/yield: '))
    cut = raw_input('Cut (Y/N): ')
    
    pullData(stock,cut)