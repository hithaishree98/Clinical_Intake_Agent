"""Tests for IdentityOut Pydantic validators (app/schemas.py)."""
import pytest
from app.schemas import IdentityOut


class TestNormPhone:
    def test_formatted_us_phone(self):
        assert IdentityOut(phone="(412) 555-1234").phone == "4125551234"

    def test_dashes(self):
        assert IdentityOut(phone="412-555-1234").phone == "4125551234"

    def test_dots(self):
        assert IdentityOut(phone="412.555.1234").phone == "4125551234"

    def test_11_digit_with_country_code(self):
        assert IdentityOut(phone="14125551234").phone == "4125551234"

    def test_country_code_with_dashes(self):
        assert IdentityOut(phone="1-412-555-1234").phone == "4125551234"

    def test_too_short_returns_empty(self):
        assert IdentityOut(phone="1234").phone == ""

    def test_non_numeric_returns_empty(self):
        assert IdentityOut(phone="not a phone").phone == ""

    def test_empty_returns_empty(self):
        assert IdentityOut(phone="").phone == ""

    def test_nine_digits_returns_empty(self):
        assert IdentityOut(phone="412555123").phone == ""


class TestNormDob:
    def test_us_slash_unambiguous(self):
        # day=15 > 12: unambiguous
        assert IdentityOut(dob="03/15/1985").dob == "1985-03-15"

    def test_iso_format_passthrough(self):
        assert IdentityOut(dob="1985-03-15").dob == "1985-03-15"

    def test_month_name_format(self):
        assert IdentityOut(dob="March 15, 1985").dob == "1985-03-15"

    def test_ordinal_suffix_stripped(self):
        assert IdentityOut(dob="15th March 1985").dob == "1985-03-15"

    def test_ambiguous_numeric_returns_empty(self):
        # 01 and 02 are both valid months — ambiguous; identity_node will re-ask
        assert IdentityOut(dob="01/02/1990").dob == ""

    def test_same_month_and_day_not_ambiguous(self):
        # 01/01/1990: a == b, so the guard (a != b) doesn't fire
        assert IdentityOut(dob="01/01/1990").dob == "1990-01-01"

    def test_day_over_12_not_ambiguous(self):
        # 13 > 12: only valid as DD, not ambiguous
        assert IdentityOut(dob="13/01/1990").dob == "1990-01-13"

    def test_unknown_placeholder_returns_empty(self):
        assert IdentityOut(dob="unknown").dob == ""

    def test_na_placeholder_returns_empty(self):
        assert IdentityOut(dob="n/a").dob == ""

    def test_unparseable_garbage_returns_empty(self):
        assert IdentityOut(dob="not a date").dob == ""

    def test_empty_returns_empty(self):
        assert IdentityOut(dob="").dob == ""


class TestNormName:
    def test_lowercase_title_cased(self):
        assert IdentityOut(name="jane doe").name == "Jane Doe"

    def test_all_caps_title_cased(self):
        assert IdentityOut(name="JOHN SMITH").name == "John Smith"

    def test_multi_word_name(self):
        assert IdentityOut(name="mary ann jones").name == "Mary Ann Jones"

    def test_unknown_placeholder_returns_empty(self):
        assert IdentityOut(name="unknown").name == ""

    def test_na_placeholder_returns_empty(self):
        assert IdentityOut(name="n/a").name == ""

    def test_none_placeholder_returns_empty(self):
        assert IdentityOut(name="none").name == ""

    def test_not_provided_returns_empty(self):
        assert IdentityOut(name="not provided").name == ""

    def test_empty_returns_empty(self):
        assert IdentityOut(name="").name == ""


class TestStripAddress:
    def test_full_address_with_zip_passes(self):
        addr = "123 Main St, Pittsburgh PA 15213"
        assert IdentityOut(address=addr).address == addr

    def test_address_without_zip_returns_empty(self):
        assert IdentityOut(address="123 Main St, Pittsburgh PA").address == ""

    def test_zip_plus_four_accepted(self):
        addr = "123 Main St, Pittsburgh PA 15213-1234"
        assert IdentityOut(address=addr).address == addr

    def test_unknown_placeholder_returns_empty(self):
        assert IdentityOut(address="unknown").address == ""

    def test_empty_returns_empty(self):
        assert IdentityOut(address="").address == ""
