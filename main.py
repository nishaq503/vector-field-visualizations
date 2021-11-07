import os

with open('.github_token') as infile:
    os.environ['GITHUB_AUTH'] = infile.readline()


if __name__ == '__main__':
    print('hello world')
