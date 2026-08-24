try:
    with open('data.txt','r')as f:
        content=f.read()
        
except FileNotFoundError as fe:
    print("The message is that",fe)
except Exception as ex:
    print(ex)
else:
    print("Else Statement")
finally:
    print("Must be printed")