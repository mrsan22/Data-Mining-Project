import math
def gini(p,n):
    gini=1-(math.pow((p/(p+n)),2))-(math.pow((n/(p+n)),2))
    return (gini)
if __name__=="__main__":
    x=int(input("enter total rows in original table"))
    t=int(input("enter rows in subset table"))
    
    p1=int(input("enter p of 1st subset table"))
    n1=int(input("enter n of 1st subset table"))
    p2=int(input("enter p of 2nd subset table"))
    n2=int(input("enter n of 2nd subset table"))
    gini=t/x*(gini(p1,n1))+((x-t)/x)*(gini(p2,n2))
    print("GINIa(D):")
    print(gini)
w=int(input())
    
