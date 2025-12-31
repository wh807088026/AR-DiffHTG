import random
import re

def extract_words(file_path, num_samples=500, output_file='selected_words.txt'):
    # 存储提取的单词
    words = set()  # 使用集合来自动去除重复的单词

    # 打开文件并读取每一行
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行尾的换行符并分割每一行
            parts = line.strip().split(' ')
            if len(parts) > 1:
                text = parts[-1]

                # 仅保留字母字符并分割为单词
                clean_text = re.sub(r'[^a-zA-Z\s]', '', text)  # 去除非字母字符和标点符号
                words_in_line = clean_text.split()

                # 过滤掉包含非字母的单词
                words_in_line = [word for word in words_in_line if word.isalpha()]

                # 将符合条件的单词添加到 words 集合中
                words.update(words_in_line)

    # 如果单词数不足500个，继续从剩余的单词中随机挑选直到达到所需数量
    if len(words) < num_samples:
        print(f"提取的单词数量不足 {num_samples} 个，当前总共有 {len(words)} 个唯一单词。")
        num_samples = len(words)  # 如果不足500个，则取所有唯一单词

    # 随机挑选 num_samples 个单词
    selected_words = random.sample(words, num_samples)

    # 保存到文件
    with open(output_file, 'w') as output:
        for word in selected_words:
            output.write(word + '\n')

    print(f"已将 {num_samples} 个随机单词保存到 {output_file} 文件中。")

# 示例文件路径
file_path = 'IAM64_train.txt'
output_file = 'in_words.txt'

# 提取并保存500个随机单词
extract_words(file_path, num_samples=500, output_file=output_file)
