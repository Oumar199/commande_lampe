import time
import concurrent.futures

t1 = time.perf_counter()

a = 0
def while_test(process):
    global a
    if process == 0:
        while True:
            a += 0.01
            if a > 10000000:
                break
    elif process == 1:
        while True:
            exit = input("Voulez-vous quitter ! q si oui")
            print(a)
            if exit == "q":
                break

def main():

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(while_test, [0, 1])

    # for img_name in img_names:
    #     img = Image.open(img_name)

    #     img = img.filter(ImageFilter.GaussianBlur(15))

    #     img.thumbnail(size)
    #     img.save(f'processed/{img_name}')
    #     print(f'{img_name} was processed...')


    t2 = time.perf_counter()

    print(f'Finished in {t2-t1} seconds')
    
if __name__=="__main__":
    main()
