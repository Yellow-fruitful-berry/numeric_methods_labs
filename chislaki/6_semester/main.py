LN_TEN = 2.302851
def taylor(value, eps=0.000001):

    n = 1
    value_copy = value
    power = 0
    if(value_copy < 1) :
        while (value_copy < 0.1):
            value_copy = value_copy * 10
            power -= 1
            value = value * 10
    else:
        while(value_copy > 1):
            value_copy = value_copy // 10
            power += 1
            value = value / 10

    value -= 1

    current = value  # a_n = x
    result = 0
    #print (n, " ", current, " ", result)
    while(abs(current/n) >= eps):
        result += current / n
        n += 1
        current = value*current*(-1)
        #print(n, " ", current/n, " ", result)

    result = result / LN_TEN
    #print(result, power)
    result += power
    #print(result)

    return result


num = float(input())
print(taylor(num))


# lg 5 = 2.23
