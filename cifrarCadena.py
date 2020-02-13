def convertir(text,desp):

    resul = ""

    for car in text: #Itera cada letra del texto

        if car.isalpha(): #Si es carácter, se sustituye la letra

            if car.islower():

                posOriginal = ord(car)
                print("\nCaracter: ",car)
                print("Posición original: ",posOriginal)
                print("Desplazamiento: ",desp)
                despINT = int(desp)
                print("Desplazamiento INT: ",despINT)
                
                print("Tipo de datos de 'caracter': ",type(car))
                print("Tipo de datos de 'Posición original': ",type(posOriginal))
                print("Tipo de datos de 'desplazamiento': ",type(desp))
                print("Tipo de datos de 'desplazamiento INT': ",type(despINT))
                
                posCifrada = posOriginal + despINT
                print("Posición cifrada': ",posCifrada)
                
                caracterCifrado = chr(posCifrada)
                
                
                resul += chr((ord(car) - 97 + int(desp)) % 26 + 97)
                print("resul': ",resul)
               

            if car.isupper():
    
                resul += chr((ord(car) - 65 + int(desp)) % 26 + 65)   

        else: #No se sustituyen otros símbolos, pertecen igual

            resul += car

    return resul   

if __name__ == "__main__":   # programa principal

    texto = input("Texto a cifrar: ")

    desp = input("Desplazamiento letra: ")

    if desp.isdigit():

        cifrado = convertir(texto,desp)

        print (cifrado)

    else:

        print ("El desplazamiento ha de ser un digito")