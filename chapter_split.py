from word2number import w2n
import re
import textwrap

# Function for convert text numbers to numbers
def convert_word_to_number(word):
    try:
        number = w2n.word_to_num(word)
    except:
        pass
        number = None
    return number

def split_chapters(wrapped_text):
  try:
    hole_para = wrapped_text.strip()
    chapter_list = []
    output_chapter_text = ""
    book_dict = {}

    # Check it is introduction
    if re.compile(r'\b' + re.escape("introduction") + r'\b', re.IGNORECASE).search(wrapped_text[:100]):
      lines = wrapped_text.split('.')
      modified_text = '.'.join(lines[1:])
      output_chapter_text += "----------Introduction-----------"
      output_chapter_text += "\n"
      output_chapter_text += modified_text
      book_dict = {"Introduction": modified_text}
      return book_dict

    # Split chapter text
    word = 'chapter'
    split_text = re.split(rf'(?i)\s*{re.escape(word)}\s*', hole_para)

    for idx, text in enumerate(split_text):
      if text.strip() != "" and len(text) > 1:
        words = text.split()
        first_word = words[0]
        last_word = words[len(words) -1]
        chapter_numb = convert_word_to_number(first_word[:-1])

        if chapter_numb is not None and chapter_numb not in chapter_list:
          # Append the chapter number incrementally
          #print("\n")
          chapter_list.append(chapter_numb)
          output_chapter_text += f"CHAPTER."
          #print("----------------", "Chapter:", chapter_numb, "------------------")
          # Remove Chapter numbers in sentences
          words = words[1:]
          # Captialize first letter in first word
          word_one  = words[0].capitalize()

          # Remove End word in the sentences
          word_lst = words[len(words) - 1].lower()

          if word_lst == "and" or word_lst == "end":
            words.pop()

          # Generate the sentence
          chapter_text = f"{word_one} " + " ".join(words[1:])
          chapter_text = textwrap.fill(chapter_text.strip(), width=80)
          chapter_text = chapter_text.strip()
          output_chapter_text += chapter_text
          #print(chapter_text)

        else:
          word_one  = words[0]
          if word_one[len(word_one) -1 ] == "." or word_one == ".":
            words = words[1:]
          else:
            words = ["chapter "] + words

          # Remove End word in the sentences
          try:
            word_lst = words[len(words) - 1].lower()
            if word_lst == "and" or word_lst == "end":
              words.pop()
          except:
            pass

          chapter_text = " ".join(words)
          chapter_text = textwrap.fill(chapter_text.strip(), width=80)
          chapter_text = chapter_text.strip()
          output_chapter_text += chapter_text

    split_chapters = output_chapter_text.split("CHAPTER.")
    chapter_count = 0
    for chapter in split_chapters:
        if len(chapter) > 0:
          book_dict[f'Chapter {chapter_list[chapter_count]}'] = chapter
          chapter_count += 1

    return book_dict

    #return output_chapter_text
  except Exception as e:
    return f"Error occured when coverting to the chapters: {e}"