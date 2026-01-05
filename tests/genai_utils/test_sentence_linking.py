from pytest import mark, param

from genai_utils.models import Span
from genai_utils.sentence_linking import (
    find_quote_in_sentence,
    get_best_matching_sentence_for_quote,
    link_quotes_and_sentences,
)


@mark.parametrize(
    "sentence,claim,expected",
    [
        param(
            "claim in sentence",
            "claim in sentence",
            Span(start=0, end=17, text="claim in sentence"),
            id="perfect match",
        ),
        param(
            "the claim in sentence nicely",
            "claim in sentence",
            Span(start=4, end=21, text="claim in sentence"),
            id="claim in sentence",
        ),
        param(
            "claim in sentence",
            "the claim in sentence nicely",
            Span(start=0, end=17, text="claim in sentence"),
            id="sentence in claim",
        ),
        param(
            "Mr Blobby was in his usual fine form tonight on top of the pops.",
            "Mr Blobby was in his usual fine form tonight on topofthepops",
            Span(
                start=0,
                end=60,
                text="Mr Blobby was in his usual fine form tonight on top of the p",
            ),
            id="almost perfect match",
        ),
        param(
            "this text does not match the other",
            "no relation at all",
            None,
            id="no match",
        ),
        param(
            "Mr Johnson said, 'I really am a VERY CLEVER boy indeed!'.",
            '"I really am a very clever boy indeed"',
            Span(start=18, end=56, text="I really am a VERY CLEVER boy indeed!'"),
            id="punctuation change",
        ),
        param(
            "A sentence with a tpyo in it.",
            "A sentence with a typo in it.",
            Span(start=0, end=29, text="A sentence with a tpyo in it."),
            id="typo",
        ),
        param(
            "والثلاثاء, تم توقيف رجل يشتبه بارتكابه الجريمة في سويسرا, بحسب مسؤولين فرنسيين.",
            'وأفاد مكتب المدعي العام في باريس الثلاثاء بأنه تم التعرّف على هوية المشتبه به "وتوقيفه اليوم في كانتون جنيف".',
            None,
            id="Arabic no match",
        ),
        param(
            "واستمع المعايطة إلى ملاحظات رؤساء اللجان وأهم التحديات التي واجهتهم في جميع مراحل العملية الانتخابية, كما وسيتم تقديم توصيات مكتوبة من قبل لجان الانتخاب, لتتم مناقشتها ودراستها من خلال عدة جلسات, للاستفادة منها في تجويد العملية الانتخابية مستقبلا.",
            "وستتم تقديم توصيات مكتوبة من قبل لجان الانتخاب, لتتم مناقشتها ودراستها من خلال عدة جلسات, للاستفادة منها في تجويد العملية الانتخابية مستقبلا.",
            Span(
                start=106,
                end=247,
                text="وسيتم تقديم توصيات مكتوبة من قبل لجان الانتخاب, لتتم مناقشتها ودراستها من خلال عدة جلسات, للاستفادة منها في تجويد العملية الانتخابية مستقبلا.",
            ),
            id="Arabic good match",
        ),
        param(
            "Cependant, n'ayant pas assez d'espace dans sa demeure pour contenir plus de 50 candidats, Lamine Gaye se charge de les transporter au domicile de Fatou Sall qui se trouve être la grande sœur de Pape Sow, l'un des plus proches collaborateurs du lutteur.",
            "Fatou Sall qui se trouve être la grande sœur de Pape Sow, l'un des plus proches collaborateurs du lutteur.",
            Span(
                start=146,
                end=252,
                text="Fatou Sall qui se trouve être la grande sœur de Pape Sow, l'un des plus proches collaborateurs du lutteur.",
            ),
            id="French good match",
        ),
    ],
)
def test_find_claim_in_sentence(sentence: str, claim: str, expected: Span) -> None:
    span = find_quote_in_sentence(sentence, claim)
    assert span == expected


@mark.parametrize(
    "quote,sentences,expected_sent_idx",
    [
        param(
            "I will not seek reelection",
            [
                "In a speech Biden indicated he would not run again.",
                '"I will not seek reelection", said the President to a room of journalists',
            ],
            1,
            id="simple example",
        ),
        param(
            """"I think that we're stuck in that war unless I'm president. I'll get it done. I'll negotiate; I'll get us out. We gotta get out. Biden says, 'We will not leave until we win,'" Trump argued.""",
            [
                """\"I think that we're stuck in that war unless I'm president. I'll get it done. I'll negotiate; I'll get us out. We gotta get out. Biden says, 'We will not leave until we win,'\"""",
                "Trump argued.",
            ],
            0,
            id="Tiny second sentence within quote.",
        ),
        param(
            "'The information sharing with us was not adequate, it was worse than that, as it was basically non-existent. Everything I've since learned about what happened I've learned through the police, the trial and my solicitors.",
            [
                "This is an added sentence before the useful ones",
                "'The information sharing with us was not adequate, it was worse than that, as it was basically non-existent.",
                "Everything I've since learned about what happened I've learned through the police, the trial and my solicitors.",
                "And a final one for good measure.",
            ],
            2,
            id="multiple sentences in claim",
        ),
        param(
            "سرايا - أعلن ,اليوم الاربعاء, عن شواغر وظيفية, وكما دعت مؤسسات مرشحين للحضور لغاية المقابلة الشخصية استكمال اجراءات التعيين.",
            [
                "سرايا - أعلن ,اليوم الاربعاء, عن شواغر وظيفية, وكما دعت مؤسسات مرشحين للحضور لغاية المقابلة الشخصية استكمال اجراءات التعيين.",
                "وتاليا التفاصيل والأسماء:",
            ],
            0,
            id="Arabic example",
        ),
        param(
            'وأكدت جلالتها على ان "للولايات المتحدة نفوذاً عسكرياً واقتصادياً ودبلوماسياً يمكنها استخدامه مع إسرائيل، وأن عليها البدء في استخدامه، "لأن مخاطر التصعيد مرتفعة جداً الآن."',
            [
                'وقالت حان الوقت ليتحرك المجتمع الدولي، مشيرة الى "أن التعبير عن القلق أو حتى الدعوات إلى وقف إطلاق النار لا معنى لها طالما يتم الاستمرار في إمداد الأسلحة التي تقتل المدنيين".',
                'وأكدت جلالتها على ان "للولايات المتحدة نفوذاً عسكرياً واقتصادياً ودبلوماسياً يمكنها استخدامه مع إسرائيل، وأن عليها البدء في استخدامه،."',
                '"لأن مخاطر التصعيد مرتفعة جداً الآن."'
                'وأشارت إلى ان السبب الجذري لهذا الصراع لم يبدأ في السابع من تشرين الأول، وبينت في نهاية المقابلة ان فشل محادثات السلام في الماضي كان بسبب عدم بذل أي جهد لتطبيق القانون الدولي ولعدم وضع كلف أو عواقب لردع الاحتلال، "لذلك شعرت إسرائيل بالاستقواء وقامت ببناء المزيد من المستوطنات، والاستيلاء على المزيد من الأراضي".',
            ],
            1,
            id="Arabic harder example",
        ),
        param(
            """Au Sénégal, son pays natal, on lui reconnait cet attribut au regard de son parcours à travers le profond hinterland du pays, avant et post indépendance, pour installer les bases du système éducatif alors embryonnaire, dans des conditions qu'il ne partageait alors qu'avec " les médecins de campagne ".""",
            [
                """Au Sénégal, son pays natal, on lui reconnait cet attribut au regard de son parcours à travers le profond""",
                "hinterland",
                """du pays, avant et post indépendance, pour installer les bases du système éducatif alors embryonnaire, dans des conditions qu'il ne partageait alors qu'avec " les médecins de campagne ".""",
            ],
            2,
            id="French example",
        ),
        param(
            "Claim not in sentences",
            ["Nothing to see here.", "Nowt here either."],
            -1,
            id="No sentences matching claim",
        ),
    ],
)
def test_get_best_matching_sentence_for_quote(
    quote: str,
    sentences: list[str],
    expected_sent_idx: int,
) -> None:
    best_match = get_best_matching_sentence_for_quote(quote, sentences)

    if best_match is None:
        assert expected_sent_idx == -1
        return

    assert best_match[0] == expected_sent_idx  # check it links to the right sentence
    assert best_match[1] is not None  # check the span got added successfully


@mark.parametrize(
    "quotes,sentences,expected",
    [
        param(
            [
                "he had seen mr smith in the shop",
                "denies all charges",
            ],  # quotes
            [
                "the witness said he had seen mr smith in the shop",
                "mr smith denies all charges",
            ],  # sentences
            [
                (
                    0,
                    0,
                    Span(
                        start=17,
                        end=49,
                        text="he had seen mr smith in the shop",
                    ),
                ),
                (1, 1, Span(start=9, end=27, text="denies all charges")),
            ],
            id="basic example",
        ),
        param(
            [
                "وقالت حان الوقت ليتحرك المجتمع الدولي،",
                " وقف إطلاق النار لا معنى لها طالما يتم الاستمرار في إمداد الأسلحة التي تقتل المدنيين",
                'وأكدت جلالتها على ان "للولايات المتحدة نفوذاً عسكرياً واقتصادياً ودبلوماسياً يمكنها استخدامه مع إسرائيل، وأن عليها البدء في استخدامه، "لأن مخاطر التصعيد مرتفعة جداً الآن."',
                "وقالت حان الوقت ليتحرك المجتمع الدولي،",
                '"لذلك شعرت إسرائيل بالاستقواء وقامت ببناء المزيد من المستوطنات، والاستيلاء على المزيد من الأراضي".',
                "ادعاء ليس له مكان هنا، لا يتطابق مع هذا",
            ],  # quotes
            [
                'وقالت حان الوقت ليتحرك المجتمع الدولي، مشيرة الى "أن التعبير عن القلق أو حتى الدعوات إلى وقف إطلاق النار لا معنى لها طالما يتم الاستمرار في إمداد الأسلحة التي تقتل المدنيين".',
                'وأكدت جلالتها على ان "للولايات المتحدة نفوذاً عسكرياً واقتصادياً ودبلوماسياً يمكنها استخدامه مع إسرائيل، وأن عليها البدء في استخدامه،."',
                '"لأن مخاطر التصعيد مرتفعة جداً الآن."',
                'وأشارت إلى ان السبب الجذري لهذا الصراع لم يبدأ في السابع من تشرين الأول، وبينت في نهاية المقابلة ان فشل محادثات السلام في الماضي كان بسبب عدم بذل أي جهد لتطبيق القانون الدولي ولعدم وضع كلف أو عواقب لردع الاحتلال، "لذلك شعرت إسرائيل بالاستقواء وقامت ببناء المزيد من المستوطنات، والاستيلاء على المزيد من',
                ' الأراضي".',
                "جملة إضافية في النهاية ليس لها أي صلة.",
            ],  # sentences
            [
                (
                    0,
                    0,
                    Span(
                        start=0,
                        end=38,
                        text="وقالت حان الوقت ليتحرك المجتمع الدولي،",
                    ),
                ),
                (
                    1,
                    0,
                    Span(
                        start=88,
                        end=172,
                        text=" وقف إطلاق النار لا معنى لها طالما يتم الاستمرار في إمداد الأسلحة التي تقتل المدنيين",
                    ),
                ),
                (
                    2,
                    1,
                    Span(
                        start=0,
                        end=135,
                        text='وأكدت جلالتها على ان "للولايات المتحدة نفوذاً عسكرياً واقتصادياً ودبلوماسياً يمكنها استخدامه مع إسرائيل، وأن عليها البدء في استخدامه،."',
                    ),
                ),
                (
                    3,
                    0,
                    Span(
                        start=0,
                        end=38,
                        text="وقالت حان الوقت ليتحرك المجتمع الدولي،",
                    ),
                ),
                (
                    4,
                    3,
                    Span(
                        start=213,
                        end=301,
                        text='"لذلك شعرت إسرائيل بالاستقواء وقامت ببناء المزيد من المستوطنات، والاستيلاء على المزيد من',
                    ),
                ),
            ],  # expected
            id="Arabic example",
        ),
        param(
            [
                "conditions qu'il ne partageait alors qu'avec « les médecins de campagne »",
                "N'apparaît pas dans le texte.",
                "y compris lorsqu'il s'est agi pour lui de se mettre sous le drapeau de la France, pour aller combattre lors de la seconde guerre mondiale",
                "l'engagement militant pour son pays ne l'ont jamais quitté",
            ],  # quotes
            [
                "Au Sénégal, son pays natal, on lui reconnait cet attribut au regard de son parcours à travers le profond hinterland du pays, avant et post indépendance, pour installer les bases du système éducatif alors embryonnaire, dans des conditions qu'il ne partageait alors qu'avec « les médecins de campagne ».",
                "Cette vocation d'enseignant et de pédagogue hors pairs, et l'engagement militant pour son pays ne l'ont jamais quitté. Ainsi à tous les postes qu'il a eu à occuper le Président Amadou Mahtar Mbow a servi avec fierté et dévouement, y compris lorsqu'il s'est agi pour lui de se mettre sous le drapeau de la France, pour aller combattre lors de la seconde guerre mondiale, avec la ferme conviction de revenir servir son pays le Sénégal, dont il pensait qu'il en avait besoin.",
                "Avec le recul, on se rend compte aujourd'hui, qu'Amadou Mahtar Mbow nous a fait la preuve d'un altruisme extraordinaire qui a surpris plus d'un. ",
            ],  # sentences
            [
                (
                    0,
                    0,
                    Span(
                        start=227,
                        end=300,
                        text="conditions qu'il ne partageait alors qu'avec « les médecins de campagne »",
                    ),
                ),
                (
                    2,
                    1,
                    Span(
                        start=231,
                        end=368,
                        text="y compris lorsqu'il s'est agi pour lui de se mettre sous le drapeau de la France, pour aller combattre lors de la seconde guerre mondiale",
                    ),
                ),
                (
                    3,
                    1,
                    Span(
                        start=59,
                        end=117,
                        text="l'engagement militant pour son pays ne l'ont jamais quitté",
                    ),
                ),
            ],  # expected
            id="French example",
        ),
    ],
)
def test_link_quotes_and_sentences(
    quotes: list[str],
    sentences: list[str],
    expected: list[tuple[int, int, Span]],
) -> None:
    claimants_per_sentence = link_quotes_and_sentences(quotes, sentences)
    assert claimants_per_sentence == expected


@mark.parametrize(
    "quotes,sentences,threshold,passes",
    [
        param(
            [
                "I will not seek reelection",
                "I won't seek reelection",
                "No reelection for me",
            ],
            [
                "I will not seek reelection",
            ],
            70,
            [True, True, True],
            id="low threshold",
        ),
        param(
            [
                "I will not seek reelection",
                "I won't seek reelection",
                "No reelection for me",
            ],
            [
                "I will not seek reelection",
            ],
            80,
            [True, True, False],
            id="mid threshold",
        ),
        param(
            [
                "I will not seek reelection",
                "I won't seek reelection",
                "No reelection for me",
            ],
            [
                "I will not seek reelection",
            ],
            100,
            [True, False, False],
            id="high threshold",
        ),
    ],
)
def test_threshold_is_passed_along(
    quotes: list[str], sentences: list[str], threshold: float, passes: list[bool]
):
    linked = link_quotes_and_sentences(quotes, sentences, threshold)

    quotes_matched = [m[0] for m in linked]
    for i, should_pass in enumerate(passes):
        if should_pass:
            assert i in quotes_matched
        else:
            assert i not in quotes_matched
