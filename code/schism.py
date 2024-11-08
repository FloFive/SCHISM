def show_choices():
    print("Select the action you want to perform:")
    print("Select 1 for training:")
    print("Select 2 for inference:")
    print("3. Option 3")
    print("4. Quitter")


def option_training():
    print("The action performed is : Training.")
    # select the hyperparameters
    # select the images and scanners
    # Launch Training and get weights in return


def option_inference():
    print("The action performed is : inférence.")
    # select the hyperparameters
    # select the images and weights
    # Launch inference and get segmentation in return
    # show results as graph


def option_3():
    print("Your choice is 3.")


def main():
    while True:
        show_choices()
        choice = input("Make a choice between 1 to 4: ")

        if choice == '1':
            option_training()
        elif choice == '2':
            option_inference()
        elif choice == '3':
            option_3()
        elif choice == '4':
            print("Leaving SCHISM !")
            break
        else:
            print("Your choice is not recognised !")


if __name__ == "__main__":
    main()
