"""Reading a lead list the way it actually arrives.

Lead lists come from whichever agency sent them last: a header row that may not be the first
row, a phone column called "Mobile" or "Contact No.", numbers Excel has stored as floats, blank
rows in the middle, and the same person twice. An importer that rejects the file because row
400 of 3,000 is wrong is an importer nobody can use — they have no way to find row 400.

parse() touches no database on purpose, so the preview an operator confirms and the rows that
are then written come from the same code. A preview computed differently to the commit is a
preview that lies.
"""

import io

import pytest
from openpyxl import Workbook

from app.services.contact_import import (
    MAX_IMPORT_ROWS,
    ImportReport,
    find_columns,
    parse,
)


def sheet(rows) -> io.BytesIO:
    workbook = Workbook()
    ws = workbook.active
    for row in rows:
        ws.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    return buffer


def numbers(report: ImportReport):
    return [c.phone_number for c in report.contacts]


# --- finding the columns --------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    ["Phone", "Mobile", "Contact No.", "MOBILE NUMBER", "phone number", "WhatsApp", "Cell"],
)
def test_the_phone_column_is_found_by_whatever_it_is_called(header):
    """Every one of these is a real spelling from a real export."""
    report = parse(sheet([("Name", header), ("Rahul", "9876543210")]))
    assert numbers(report) == ["+919876543210"]


def test_a_title_row_above_the_header_does_not_break_it():
    """Exports commonly carry a title and a blank line before the header."""
    report = parse(
        sheet([("Lead list — August", None), (None, None), ("Name", "Phone"), ("Amit", "9876543213")])
    )
    assert report.header_row == 3
    assert numbers(report) == ["+919876543213"]


def test_a_file_with_no_header_at_all_still_imports():
    """Found by the shape of the data instead: a column where most values have ten or more
    digits is the phone column whatever it is called."""
    report = parse(sheet([("Sunil", "9876543214"), ("Meena", "9876543215")]))
    assert report.header_row is None
    assert numbers(report) == ["+919876543214", "+919876543215"]
    assert [c.name for c in report.contacts] == ["Sunil", "Meena"]


def test_a_pincode_column_is_not_mistaken_for_a_phone_number():
    """Six digits is not enough to dial, and picking the wrong column silently dials nothing."""
    report = parse(sheet([("Name", "Pincode", "Mobile"), ("Rahul", "560103", "9876543210")]))
    assert report.phone_column == "C"
    assert numbers(report) == ["+919876543210"]


def test_a_headerless_sheet_picks_the_dialable_column_not_the_pincode():
    """With headers the names decide it. Without them only the digit count does, and six
    digits looking dialable means every call goes to a pincode."""
    report = parse(sheet([("Rahul", "560103", "9876543210"), ("Asha", "560037", "9876543211")]))
    assert report.phone_column == "C"
    assert numbers(report) == ["+919876543210", "+919876543211"]


def test_no_phone_column_is_reported_rather_than_guessed():
    report = parse(sheet([("Name", "City"), ("Rahul", "Bengaluru")]))
    assert report.contacts == []
    assert report.problems
    assert "phone" in report.problems[0].reason.lower()


def test_an_empty_sheet_reports_a_problem_and_no_contacts():
    report = parse(sheet([]))
    assert report.contacts == []
    assert report.problems


def test_the_columns_are_reported_back():
    """A name column read as the phone column dials nothing; a phone column read as the name
    column makes the agent greet somebody as "9876543210". The operator has to be able to see
    which was which before committing."""
    report = parse(sheet([("Name", "Mobile"), ("Rahul", "9876543210")]))
    assert (report.phone_column, report.name_column, report.header_row) == ("B", "A", 1)


def test_the_primary_number_wins_over_an_alternate_one():
    """A sheet carrying both is ordinary. Taking the leftmost match picks "Alt Number", and
    then every call goes to the line the prospect gave as a fallback."""
    header, phone, _ = find_columns([("Alt Number", "Mobile"), ("9876543210", "9999999999")])
    assert (header, phone) == (0, 1)


def test_the_specific_spellings_outrank_the_vague_ones():
    """The ordering IS the decision, so it is asserted directly. Through find_columns it is
    masked by the secondary-marker penalty: reorder the tuple and "Alt Number" still loses,
    because of "alt" rather than because of the ranking."""
    from app.services.contact_import import _PHONE_HEADERS

    order = {token: i for i, token in enumerate(_PHONE_HEADERS)}
    for vague in ("number", "contact", "tel"):
        for specific in ("mobile", "phone"):
            assert order[specific] < order[vague], (
                f"'{vague}' outranks '{specific}', so a column merely called "
                f"'{vague}' wins over the one that says '{specific}'"
            )


def test_an_alternate_number_is_still_used_when_it_is_the_only_one():
    """Losing to a primary column is not the same as being rejected."""
    report = parse(sheet([("Name", "Alt Mobile"), ("Rahul", "9876543210")]))
    assert numbers(report) == ["+919876543210"]


@pytest.mark.parametrize("secondary", ["Alternate Phone", "Secondary Mobile", "Office Number"])
def test_secondary_columns_lose_to_the_primary(secondary):
    header, phone, _ = find_columns([(secondary, "Mobile"), ("9876543210", "9999999999")])
    assert phone == 1


# --- what a row becomes ---------------------------------------------------------------


def test_a_numeric_cell_is_read_as_its_digits():
    """openpyxl hands an integral numeric cell back as int, so this exercises that path rather
    than the float one — which is exactly why the float branch needed its own test below."""
    report = parse(sheet([("Name", "Phone"), ("Vikas", 9876543216.0)]))
    assert numbers(report) == ["+919876543216"]


def test_a_fractional_cell_is_rejected_rather_than_run_together():
    """The dangerous case, and it was silently wrong. repr(9876543217.5) is "9876543217.5";
    the dot is stripped as ordinary punctuation and the digits either side run together into
    +9198765432175 — a valid-looking number belonging to somebody else."""
    report = parse(sheet([("Name", "Phone"), ("Asha", 9876543217.5)]))
    assert report.contacts == []
    assert "decimal" in report.problems[0].reason


def test_the_fraction_check_is_what_stops_it():
    """Direct, because the value has to survive openpyxl to reach the parser at all."""
    from app.services.contact_import import _cell_to_text, is_fraction

    assert is_fraction(9876543217.5)
    assert not is_fraction(9876543216.0)
    assert not is_fraction(9876543216)
    # And the text kept for the error message still shows the operator what was in the cell.
    assert _cell_to_text(9876543217.5) == "9876543217.5"


def test_scientific_notation_is_reported_and_not_guessed_at():
    """Excel renders a long number as 9.87654E+09 and the digits are genuinely gone. Inventing
    them would dial a stranger."""
    report = parse(sheet([("Name", "Phone"), ("Vikas", "9.8765E+09")]))
    assert report.contacts == []
    assert "scientific notation" in report.problems[0].reason
    assert "Text" in report.problems[0].reason, "the operator is not told how to fix it"


@pytest.mark.parametrize(
    "raw", ["98765 43210", "098765-43210", "+91 98765 43210", "(98765) 43210", "91 9876543210"]
)
def test_punctuation_the_operator_never_chose_is_accepted(raw):
    report = parse(sheet([("Name", "Phone"), ("Rahul", raw)]))
    assert numbers(report) == ["+919876543210"]


def test_a_bad_number_costs_its_own_row_and_not_the_file():
    """The whole point. 3,000 rows must not be lost to one bad cell."""
    report = parse(
        sheet([("Name", "Phone"), ("A", "9876543217"), ("B", "not a number"), ("C", "9876543218")])
    )
    assert numbers(report) == ["+919876543217", "+919876543218"]
    assert [p.row for p in report.problems] == [3]


def test_a_rejected_row_names_its_line_number():
    """An operator fixing the source file needs the row, not the phone number."""
    report = parse(sheet([("Name", "Phone"), ("A", "9876543217"), ("B", "xyz")]))
    assert report.problems[0].row == 3
    assert report.problems[0].value == "xyz"


def test_a_duplicate_inside_the_file_is_reported_against_the_first_row():
    report = parse(
        sheet([("Name", "Phone"), ("A", "9876543217"), ("B", "98765 43217")])
    )
    assert numbers(report) == ["+919876543217"]
    assert "row 2" in report.problems[0].reason


def test_blank_rows_are_skipped_silently():
    """A trailing blank row is a spreadsheet artefact, not something to report."""
    report = parse(
        sheet([("Name", "Phone"), ("A", "9876543217"), (None, None), (None, None)])
    )
    assert len(report.contacts) == 1
    assert report.problems == []
    assert report.total_rows == 1


def test_a_row_with_a_name_but_no_number_is_reported():
    """Silently dropping it looks like the import lost somebody."""
    report = parse(sheet([("Name", "Phone"), ("Rahul", None)]))
    assert report.contacts == []
    assert report.problems[0].value == "Rahul"


def test_a_missing_name_is_allowed():
    """Without one the agent asks for the name on the call, which is a supported path."""
    report = parse(sheet([("Name", "Phone"), (None, "9876543210")]))
    assert report.contacts[0].name is None


def test_every_import_gets_its_own_batch_id():
    """An operator who uploads the wrong file needs it gone, and picking 900 rows out by phone
    number is not something anybody will do."""
    first, second = parse(sheet([("Phone",), ("9876543210",)])), parse(sheet([("Phone",), ("9876543211",)]))
    assert first.batch_id != second.batch_id


def test_the_row_cap_stops_rather_than_truncating_silently():
    """A file over the cap must say so on the row it stopped at."""
    rows = [("Name", "Phone")] + [(f"P{i}", f"9{i:09d}") for i in range(MAX_IMPORT_ROWS + 5)]
    report = parse(sheet(rows))
    assert len(report.contacts) + len(report.problems) >= MAX_IMPORT_ROWS
    assert any(str(MAX_IMPORT_ROWS) in p.reason for p in report.problems)


def test_the_cap_is_high_enough_for_a_real_lead_list():
    assert MAX_IMPORT_ROWS >= 10_000
