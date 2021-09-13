# Airline-Price-Optimization-Micro-Challenge
Data scientists tend to focus on prediction because that's where conventional machine learning excels. But real world decision-making involves both prediction and optimization. After predicting what will happen, you decide what to do about it.

Optimization gets less attention than it deserves. So this micro-challenge will test your optimization skills as you write a function to improve how airlines set prices.

Imgur

The Problem
You recently started Aviato.com, a startup that helps airlines set ticket prices.

Aviato's success will depend on a function called pricing_function. This notebook already includes a very simple version of pricing_function. You will modify pricing_function to maximize the total revenue collected for all flights in our simulated environment.

For each flight, pricing_function will be run once per (simulated) day to set that day's ticket price. The seats you don't sell today will be available to sell tomorrow, unless the flight leaves that day.

Your pricing_function is run for one flight at a time, and it takes following inputs:

Number of days until the flight
Number of seats they have left to sell
A variable called demand_level that determines how many tickets you can sell at any given price.
The quantity you sell at any price is:

quantity_sold = demand_level - price

Ticket quantities are capped at the number of seats available.

Your function will output the ticket price.

You learn the demand_level for each day at the time you need to make predictions for that day. For all days in the future, you only know demand_level will be drawn from the uniform distribution between 100 and 200. So, for any day in the future, it is equally likely to be each value between 100 and 200.

In case this is still unclear, some relevant implementation code is shown below.

The Simulator
We will run your pricing function in a simulator to test how well it performs on a range of flight situations. Run the following code cell to set up your simulation environment:

In [1]:
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me
In case you want to check your understanding of the simulator logic, here is a simplified version of some of the key logic (leaving out the code that prints your progress). If you feel you understand the description above, you can skip reading this code.

def _tickets_sold(p, demand_level, max_qty):
        quantity_demanded = floor(max(0, p - demand_level))
        return min(quantity_demanded, max_qty)

def simulate_revenue(days_left, tickets_left, pricing_function, rev_to_date=0, demand_level_min=100, demand_level_max=200):
    if (days_left == 0) or (tickets_left == 0):
        return rev_to_date
    else:
        demand_level = uniform(demand_level_min, demand_level_max)
        p = pricing_function(days_left, tickets_left, demand_level)
        q = _tickets_sold(demand_level, p, tickets_left)
        return _total_revenue(days_left = days_left-1, 
                              tickets_left = tickets_left-q, 
                              pricing_function = pricing_function, 
                              rev_to_date = rev_to_date + p * q,
                              demand_level_min = demand_level_min,
                              demand_level_max = demand_level_max
                             )
Your Code
Here is starter code for the pricing function. If you use this function, you will sell 10 tickets each day (until you run out of tickets).

In [2]:
demand_list = []
avrg_demand = demand_list

def avrg_calc(demand_level):
    avrg_demand.append(demand_level)
    return np.mean(avrg_demand)

def std_demand(demand_level):
    return np.std(avrg_demand)
In [3]:
def pricing_function(days_left, tickets_left, demand_level):
    average_demand = avrg_calc(demand_level)
    STD_demand = std_demand(demand_level)
    price = demand_level - round((tickets_left / days_left))
    if(average_demand > demand_level and (len(avrg_demand) != 0 ) and days_left > 1 ):
        price = demand_level - (tickets_left / days_left) + 3
    if( (average_demand - STD_demand) > demand_level and (len(avrg_demand) != 0 ) and days_left > 1 ):  
        price = demand_level - (tickets_left / days_left) + 6
    if( (demand_level  >= average_demand + (1.35* STD_demand)) and (len(avrg_demand) >= 5 )):  
        price = demand_level - (tickets_left / days_left) - 11
    if( (demand_level  <= average_demand - (2* STD_demand)) and (len(avrg_demand) >= 5 )):
        price = demand_level - (tickets_left / days_left) + 9   
    return price
To see a small example of how your code works, test it with the following function:

In [4]:
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
7 days before flight: Started with 50 seats. Demand level: 131. Price set to $124. Sold 7 tickets. Daily revenue is 868. Total revenue-to-date is 868. 43 seats remaining
6 days before flight: Started with 43 seats. Demand level: 170. Price set to $163. Sold 7 tickets. Daily revenue is 1143. Total revenue-to-date is 2011. 36 seats remaining
5 days before flight: Started with 36 seats. Demand level: 185. Price set to $178. Sold 7 tickets. Daily revenue is 1248. Total revenue-to-date is 3260. 29 seats remaining
4 days before flight: Started with 29 seats. Demand level: 182. Price set to $175. Sold 7 tickets. Daily revenue is 1225. Total revenue-to-date is 4485. 22 seats remaining
3 days before flight: Started with 22 seats. Demand level: 180. Price set to $173. Sold 7 tickets. Daily revenue is 1211. Total revenue-to-date is 5695. 15 seats remaining
2 days before flight: Started with 15 seats. Demand level: 188. Price set to $180. Sold 8 tickets. Daily revenue is 1437. Total revenue-to-date is 7132. 7 seats remaining
1 days before flight: Started with 7 seats. Demand level: 183. Price set to $176. Sold 7 tickets. Daily revenue is 1230. Total revenue-to-date is 8362. 0 seats remaining
The flight took off today. 
This flight is booked full.
Total Revenue: $8362
Out[4]:
8362.380915895552
You can try simulations for a variety of values.

Once you feel good about your pricing function, run it with the following cell to to see how it performs on a wider range of flights.

In [5]:
demand_list = []
avrg_demand = demand_list
score_me(pricing_function)
Ran 200 flights starting 100 days before flight with 100 tickets. Average revenue: $17961
Ran 200 flights starting 14 days before flight with 50 tickets. Average revenue: $8212
Ran 200 flights starting 2 days before flight with 20 tickets. Average revenue: $2864
Ran 200 flights starting 1 days before flight with 3 tickets. Average revenue: $437
<IPython.core.display.Javascript object>
Average revenue across all flights is $7369
