import subprocess
import importlib
import datetime
import smtplib
import git
SMTP_PORT = 465

importlib.import_module('analyze_results')
from analyze_results import check_error

'''
Configure the system by reading from the configuration file
'''
def configure():
	build_path = "./"
	stat_path  = "./"
	confidence = 0.95
	usr        = ""
	pwd        = ""
	with open('test/statistical_testing/stat_config.txt') as config:
		lines = config.readlines()
		for line in lines:
			line_pair = line.split('=')
			if line_pair[0].rstrip() == 'build_path':
				build_path = line_pair[1].rstrip()
			elif line_pair[0].rstrip() == 'stat_path':
				stat_path = line_pair[1].rstrip()
			elif line_pair[0].rstrip() == 'confidence':
				confidence = float(line_pair[1].rstrip())
			elif line_pair[0].rstrip() == 'usr':
				usr        = line_pair[1].rstrip()
			elif line_pair[0].rstrip() == 'pwd':
				pwd        = line_pair[1].rstrip()
			else:
				print("Error: unknown configuration parameter", line_pair[0])
				exit(1)

	return build_path, stat_path, confidence, usr, pwd

'''
Run the statistical_testing executables
'''
def run_test(build_path, buffertree):
	if buffertree:
		subprocess.run(build_path + '/statistical_test', stdout=subprocess.DEVNULL, check=True)
	else:
		subprocess.run(build_path + '/mem_statistical_test', stdout=subprocess.DEVNULL, check=True)

'''
Format the results of the test and raise an error if necessary
'''
def log_result(test_name, err, err_dsc):
	if err:
		return 'ERROR Test: ' + test_name + ' = ' + err_dsc
	else:
		return 'PASSED Test: ' + test_name + ' = ' + err_dsc
'''
Send an email containing the log
'''
def send_email(err_found, log, usr, pwd):
	server_ssl = smtplib.SMTP_SSL('smtp.gmail.com', SMTP_PORT)
	server_ssl.ehlo()

	today = datetime.datetime.today()

	server_ssl.login(usr, pwd)
	subject = ''
	if err_found:
		subject = 'ERROR: '
	subject += 'Statistical Testing Log {0}/{1}/{2}'.format(str(today.month), str(today.day), str(today.year))

	msg = "\r\n".join([
		"From: "+usr,
		"To: graph.stat.testing@gmail.com",
		"Subject:"+subject,
		"",
		log
	])
	server_ssl.sendmail(usr, "graph.stat.testing@gmail.com", msg)
	server_ssl.quit()

if __name__ == "__main__":
	# Setup
	build_path, stat_path, confidence, usr, pwd = configure()
	assert usr != '' and pwd != '', "must specifiy user and password in configuration file"

	try:
		repo     = git.Repo("./")
		buf_repo = git.Repo(build_path + "/BufferTree/src/BufferTree")
	except:
		print("Must run code at root directory of StreamingRepo and must have BufferTree code present in build dir")
		exit(1)
	head = repo.heads[0]
	stream_commit_hash = head.commit.hexsha
	stream_commit_msg  = head.commit.message

	head = buf_repo.heads[0]
	buffer_commit_hash = head.commit.hexsha
	buffer_commit_msg  = head.commit.message

	log =  "StreamRepo Commit: " + stream_commit_hash + "\n" + stream_commit_msg + "\n"
	log += "BufferTree Commit: " + buffer_commit_hash + "\n" + buffer_commit_msg + "\n"

	for buffering in (True, False):
		if buffering:
			print("GutterTree")
			log += "GutterTree:\n"
		else:
			print("Standalone")
			log += "Standalone:\n"
		# Run the tests
		run_test(build_path, buffering)

		# Collect statistical results
		# test_name, test_result_file, expected_result_file
		try:
			print("small test")
			small_err, small_dsc   = check_error('small test', 'small_graph_test', stat_path + '/small_test_expected.txt')
		except Exception as err:
			small_err = True
			small_dsc = "test threw expection: {0}".format(err)
		try:
			print("medium test")
			medium_err, medium_dsc = check_error('medium test', 'medium_graph_test', stat_path + '/medium_test_expected.txt')
		except Exception as err:
			medium_err = True
			medium_dsc = "test threw expection: {0}".format(err)

		# Create a log, and send email
		log += log_result('small test', small_err, small_dsc) + "\n"
		log += log_result('medium test', medium_err, medium_dsc) + "\n"

	print("Sending email!")
	send_email(small_err or medium_err, log, usr, pwd)
