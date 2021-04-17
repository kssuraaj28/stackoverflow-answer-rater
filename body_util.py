def clear_tags(html_body):
    """
    clear the hyper links in a paragraph
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_body, features="html.parser")
    return str(soup.text)
   # for a in soup.findAll('a'):
   #     print(a)
   #     # del a['href']
   #     a.replaceWithChildren()

    # for code in soup.findAll('code'):
    #     # print(a)
    #     # del a['href']
    #     print("888888888888888888")
    #     print(code)
    #     print("888888888888888888")
    #     #code.replaceWithChildren()
    #
    #     del code

    #return str(soup)

def num_words(body):
    return len(clear_text(body).split())
