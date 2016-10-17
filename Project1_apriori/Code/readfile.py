result=[]
with open('marketing.data','r') as f:
    for line in f:
    	try:
    		result.append(list(map(int,line.split(','))))
    	except ValueError as e: print ('ValueError')
    print(result)