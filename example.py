import main_nlp

key_ans = r'''
Cardiac muscles contract automatically and rhythmically, controlled by the autonomic nervous system and specialized pacemaker cells. Intercalated Discs are unique junctions between cardiac muscle cells that contain gap junctions and desmosomes. They allow for synchronized contraction by enabling rapid electrical signal transmission and maintaining structural integrity during powerful contractions. Cardiac muscles have a striated structure similar to skeletal muscles, with alternating light and dark bands, due to the arrangement of actin and myosin filaments.
'''
stud_ans = r'''Cardiac muscles work automatically without conscious control, helping the heart pump blood continuously. They have a striated (striped) appearance due to the arrangement of actin and myosin filaments, similar to skeletal muscles. Cardiac muscles are interconnected by intercalated discs, which allow rapid transmission of electrical signals, ensuring synchronized contraction of the heart.
'''

# long ans evaluation
print(main_nlp.evaluate_paragraph(key_ans, stud_ans, 3))

# short ans evaluation
print(main_nlp.evaluate_sentence(
    'Cardiac muscles contract automatically and rhythmically controlled by the autonomic nervous system and specialized pacemaker cells', 'Cardiac muscles work automatically without conscious control, helping the heart pump blood continuously'))