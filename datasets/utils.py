import getopt
import sys

def parse_input(argv):
    batch = 256
    n_class = 0
    steps = 0
    mem_size = 0
    epochs = 0
    n_experts = 0
    k = 0

    arg_help = "{0} -b <batch> -c <n_class> -s <steps> -m <mem_size> -e <epochs> -n <n_experts> -k <k>".format(argv[0])
    
    opts, _ = getopt.getopt(argv[1:], "hb:c:s:m:e:n:k:", ["help", "batch=", "n_class=", "steps=", "mem_size=", "epochs=", "n_experts=", "k="])
    if not opts:
        print(arg_help)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-b", "--batch"):
            batch = int(arg)
        elif opt in ("-c", "--n_class"):
            n_class = int(arg)
        elif opt in ("-s", "--steps"):
            steps = int(arg)
        elif opt in ("-m", "--mem_size"):
            mem_size = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-n", "--n_experts"):
            n_experts = int(arg)
        elif opt in ("-k", "--k"):
            k = int(arg)

    return batch, n_class, steps, mem_size, epochs, n_experts, k

if __name__ == "__main__":
    batch, n_class, steps, mem_size, epochs, n_experts, k = parse_input(sys.argv)
    print("batch", batch)
    print("n_class", n_class)
    print("steps", steps)
    print("mem_size", mem_size)
    print("epochs", epochs)
    print("n_experts", n_experts)
    print("k", k)