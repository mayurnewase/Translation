import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--filename", default = "None")
parser.add_argument("--embed_dir", default = "./")
parser.add_argument("--do_preprocess", default = "True")
parser.add_argument("--lower", default = "True")
parser.add_argument("--vocab_words", default="90000")

parser.add_argument("--oov_token", default= "<unk>")
parser.add_argument("--max_seq_len", default="None")
parser.add_argument("--padding_value", default="0")
parser.add_argument("--padding_type", default= "post")
parser.add_argument("--filters", default="")

parser.add_argument("--total_size", default="160872")
parser.add_argument("--train_size", default="150000")

parser.add_argument("--d_model",  default = "128")
parser.add_argument("--d_inner",  default = "64")
parser.add_argument("--d_key",  default = "64")
parser.add_argument("--heads",  default = "5")
parser.add_argument("--layers",  default = "3")
parser.add_argument("--dropout",  default = "0.2")

parser.add_argument("--epochs",  default = "10")
parser.add_argument("--batch_size", default = "512")
parser.add_argument("--lr", default = "0.0001")

parser.add_argument("--samples",  default="1")
parser.add_argument("--log_dir",  default="logs/")
parser.add_argument("--update_freq",  default="epochs")

parser.add_argument("--files_to_copy", type = str)
parser.add_argument("--file_impo_lines", type=str)
parser.add_argument("--main_parser_lines", type=str)

parser.add_argument("--info", type=str)
parser.add_argument("--kernel_name", type=str)

args = parser.parse_args()


w = open("Singletone/singletone.py", mode = "w")
w.write("# --------- \n" + "\"\"\"\n" +  args.info + "\n\"\"\"" + "\n#-----------" +"\n\n")
for file in args.files_to_copy.split(","):
	r = open(file, "r")
	data =r.read()
	w.write("#-------- COPYING FILE " + str(file) + "--------------\n\n")
	w.write(data)
	r.close()

w.write(
	"\n"
	"filename = " +"\"" +args.filename + "\"" + "\n"
	"embed_dir = " + "\"" +args.embed_dir + "\"" + "\n"
	"do_preprocess =" + args.do_preprocess + "\n"
	
	"lower =" + args.lower + "\n"
	"vocab_words =" + args.vocab_words + "\n"
	"oov_token ="+ "\"" + args.oov_token+ "\"" + "\n"
	"max_seq_len =" + args.max_seq_len + "\n"

	"padding_value =" + args.padding_value + "\n"
	"padding_type ="+ "\"" + args.padding_type+ "\"" + "\n"
	
	"filters ="+ "\"" + args.filters+ "\""+ "\n"
	"total_size =" + args.total_size + "\n"
	"train_size =" + args.train_size + "\n"
	
	"d_model =" + args.d_model + "\n"
	"d_inner =" + args.d_inner + "\n"
	"d_key =" + args.d_key + "\n"
	"heads =" + args.heads + "\n"
	"layers =" + args.layers + "\n"
	"dropout =" + args.dropout + "\n"

	"epochs =" + args.epochs + "\n"
	"batch_size =" + args.batch_size + "\n"
	"lr =" + args.lr + "\n"

	"samples =" + args.samples + "\n"
	"log_dir ="+ "\"" + args.log_dir + "\"" + "\n"
	"update_freq ="+ "\"" + args.update_freq+ "\"" + "\n"
	)	

r = open("main.py", "r")
#data = r.read()
w.write("\n\n#-------- COPYING MAIN + --------------\n\n")

file_impo = np.arange(int(args.file_impo_lines.split(",")[0]), int(args.file_impo_lines.split(",")[1])+1)
main_parse = np.arange(int(args.main_parser_lines.split(",")[0]), int(args.main_parser_lines.split(",")[1])+1)
cut_lines = list(file_impo) + list(main_parse)
print(cut_lines)
for no,line in enumerate(r):
	if no+1 not in cut_lines:
		w.write(line)
	else:
		print(line)
	

r.close()
w.close()


w = open("Singletone/kernel-metadata.json", mode = "w")
w.write(
"{"
  "\n" 
  "\t\"id\":"+ "\""+ "mayurnewase/" + str(args.kernel_name) + "\",\n"
  "\t\"title\":"+ "\"" + str(args.kernel_name) + "\",\n"
  "\t\"code_file\":\"singletone.py\",\n"
  "\t\"language\":\"python\",\n"
  "\t\"kernel_type\":\"script\",\n"
  "\t\"is_private\":\"true\",\n"
  "\t\"enable_gpu\":\"true\",\n"
  "\t\"enable_internet\":\"true\"\n"
"}"

)
































