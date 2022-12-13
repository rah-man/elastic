inpath = "trainer_moe_imagenet_all_2_part.sh"
lines = []
with open(inpath) as f:
    lines = f.readlines()
    lines.reverse()

for line in lines:
    print(line)

outpath = "reverse.sh"
with open(outpath, "w") as f:
    for line in lines:
        f.write(line.strip() + "\n")
