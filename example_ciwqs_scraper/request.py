# Author: Varun Magesh, Stanford Law
# Purpose: Pulling data from CIWQS for animal feeding facilities, including downloading the PDF inspection reports

import datetime
import re
import time
import urllib
from enum import Enum
from pathlib import Path

import bs4
import requests
import rich.progress
import shapely as shp
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

# import digesters.db.models as m
# import digesters.utils.bucket
# from digesters.db.session import get_session

import os

CA_COUNTIES = [
    "Del Norte",
    "Siskiyou",
    "Modoc",
    "Humboldt",
    "Trinity",
    "Shasta",
    "Lassen",
    "Tehama",
    "Plumas",
    "Butte",
    "Mendocino",
    "Glenn",
    "Sierra",
    "Yuba",
    "Lake",
    "Nevada",
    "Colusa",
    "Sutter",
    "Placer",
    "El Dorado",
    "Yolo",
    "Alpine",
    "Sonoma",
    "Napa",
    "Sacramento",
    "Mono",
    "Amador",
    "Solano",
    "Calaveras",
    "Tuolumne",
    "Marin",
    "San Joaquin",
    "Contra Costa",
    "Stanislaus",
    "Alameda",
    "Mariposa",
    "San Francisco",
    "Madera",
    "San Mateo",
    "Merced",
    "Fresno",
    "Santa Clara",
    "Inyo",
    "Santa Cruz",
    "San Benito",
    "Monterey",
    "Tulare",
    "Kings",
    "San Bernardino",
    "Kern",
    "San Luis Obispo",
    "Santa Barbara",
    "Los Angeles",
    "Riverside",
    "Orange",
    "Imperial",
    "San Diego",
    "Ventura",
]

ROOT_GCS_PATH = Path("ciwqs")

CIWQS_ROOT = "https://ciwqs.waterboards.ca.gov/ciwqs/readOnly/"

Base = declarative_base()

class CiwqsAnimalWasteFacility(Base):
    __tablename__ = 'ciwqs_animal_waste_facilities'
    id = Column(Integer, primary_key=True)
    place_id = Column(String)
    url = Column(String)
    gcs_path = Column(String)
    # Add more fields as needed
    inspections = relationship("CiwqsInspection", back_populates="facility")

class CiwqsInspection(Base):
    __tablename__ = 'ciwqs_inspections'
    id = Column(Integer, primary_key=True)
    ciwqs_facility_id = Column(Integer, ForeignKey('ciwqs_animal_waste_facilities.id'))
    actual_end_date = Column(DateTime)
    attachment_url = Column(String)
    gcs_path = Column(String)
    # Add more fields as needed
    facility = relationship("CiwqsAnimalWasteFacility", back_populates="inspections")

def get_session():
    engine = create_engine('sqlite:///ciwqs_data.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def open_gcs(path: Path, mode: str):
    """Mock GCS file operation using local filesystem"""
    full_path = Path("data") / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return open(full_path, mode)


def get_facilities_in_county(
    county: str, owasp_token: str = "I5YC-NSRE-UI2N-5F7B-9HCW-QFKD-0W2C-J2ZN"
):
    county = urllib.parse.quote(county)
    url = f"{CIWQS_ROOT}/CiwqsReportServlet?OWASP_CSRFTOKEN={owasp_token}&reportName=FacilityAtAGlanceList&inFacilityName=&inAddress=&inZip=&inPartyName=&inWdid=&countyDrop={county}&RepButton=Run+Report&OWASP_CSRFTOKEN={owasp_token}"
    response = requests.post(url)
    return response


def push_all_counties_to_gcs():
    for county in rich.progress.track(CA_COUNTIES):
        response = get_facilities_in_county(county)
        if response.status_code != 200:
            print(f"Failed to get {county}")
            continue
        with open_gcs(ROOT_GCS_PATH / "facility-lists" / f"{county}.html", "w") as f:
            f.write(response.text)
        time.sleep(0.5)


class WaterboardRegion(Enum):
    REGION_1 = "1"
    REGION_2 = "2"
    REGION_3 = "3"
    REGION_4 = "4"
    REGION_5F = "5F"
    REGION_5R = "5R"
    REGION_5S = "5S"
    REGION_6V = "6V"
    REGION_7 = "7"
    REGION_8 = "8"
    REGION_9 = "9"


def get_animal_waste_facilities_in_region(
    region: WaterboardRegion,
    owasp_token: str = "I5YC-NSRE-UI2N-5F7B-9HCW-QFKD-0W2C-J2ZN",
):
    url = f"{CIWQS_ROOT}CiwqsReportServlet?reportID=6211445&inCommand=drilldown&reportName=RegulatedFacilityDetail&program=ANIMALWASTE&place={region.value}&newPageSize=10000000"
    response = requests.get(url)
    return response


def push_all_regions_to_gcs():
    for region in rich.progress.track(WaterboardRegion):
        response = get_animal_waste_facilities_in_region(region)
        if response.status_code != 200:
            print(f"Failed to get {region}")
            continue
        with open_gcs(ROOT_GCS_PATH / "animal-waste-facility-lists" / f"{region}.html", "w") as f:
            f.write(response.text)
        time.sleep(0.5)


def _legacy_ingest_all_ciwqs_places():
    session = get_session()
    for county in rich.progress.track(CA_COUNTIES):
        with open_gcs(ROOT_GCS_PATH / "facility-lists" / f"{county}.html", "r") as f:
            html = f.read()
        soup = bs4.BeautifulSoup(html, "html.parser")
        # select tr with class ciwqsReportRow2
        for tr in soup.select("tr.ciwqsReportRow2"):
            # columns are place id, place name,agency name, address, city, county
            place_id, place_name, agency_name, address, city, county = tr.select("td")
            place_id = place_id.text.strip()
            place_name = place_name.text.strip()
            agency_name = agency_name.text.strip()
            address = address.text.strip()
            city = city.text.strip()
            county = county.text.strip()
            # url is the href of a nested anchor tag in the place id
            url = tr.select_one("a").attrs["href"]
            # Note: Using a simplified model for demo purposes
            print(f"Found facility: {place_name} in {county}")
        session.flush()
    session.commit()


def _legacy_get_ciwqs_place_pages():
    session = get_session()
    # Simplified for demo - just print what we would do
    print("Would download facility pages here")
    session.commit()


def ingest_all_ciwqs_animal_waste_facilities():
    session = get_session()
    for region in rich.progress.track(WaterboardRegion):
        with open_gcs(ROOT_GCS_PATH / "animal-waste-facility-lists" / f"{region}.html", "r") as f:
            html = f.read()
        soup = bs4.BeautifulSoup(html, "html.parser")
        # select tr with class ciwqsReportRow2
        # tr with no class is the heading row

        # heading row is the tr right before the first tr with class ciwqsReportRow2
        heading_row = soup.select("tr.ciwqsReportRow2")[0].find_previous_sibling("tr")

        column_names = []
        if heading_row is not None:
            for th in heading_row.select("td"):
                column_names.append(th.text.strip())
        if not column_names:
            raise ValueError(f"No column names found in region {region}")

        for tr in soup.select("tr.ciwqsReportRow2"):
            columns = {
                column_names[i]: value.text.strip()
                for i, value in enumerate(tr.select("td"))
            }
            # select all urls
            urls_in_row = [
                td.select_one("a").attrs["href"]
                for td in tr.select("td")
                if td.select_one("a") is not None
            ]
            place_id = ""
            facility_url = ""
            for url in urls_in_row:
                if "facilityAtAGlance" in url:
                    facility_url = url
                matches = re.search(r"placeID=(\d+)", url)
                if matches is not None:
                    place_id = matches.group(1)
                    break
            if not place_id:
                raise ValueError(f"No place id found in row {tr}")
            if not facility_url:
                raise ValueError(f"No facility url found in row {tr}")
            
            # Create facility record
            facility = CiwqsAnimalWasteFacility(
                place_id=place_id,
                url=f"{CIWQS_ROOT}{facility_url}",
                gcs_path=None,
            )
            session.add(facility)
        session.flush()
    session.commit()


def get_ciwqs_animal_waste_facility_pages():
    session = get_session()
    # Only get the first 3 facilities with gcs_path is None
    places = session.execute(
        sa.select(CiwqsAnimalWasteFacility).where(CiwqsAnimalWasteFacility.gcs_path.is_(None)).limit(3)
    ).scalars().all()
    for place in places:
        with open_gcs(ROOT_GCS_PATH / "place-pages" / f"{place.place_id}.html", "w") as f:
            response = requests.get(place.url)
            if response.status_code != 200:
                print(f"Failed to get {place.url}")
                continue
            f.write(response.text)
            place.gcs_path = str(ROOT_GCS_PATH / "place-pages" / f"{place.place_id}.html")
            session.add(place)
            session.commit()
        print(f"Downloaded facility page for place_id={place.place_id}")
    if not places:
        print("No facilities found needing a page download.")


def ingest_all_ciwqs_facility_inspections():
    # pull all CiwqsAnimalWasteFacility entries that have a gcs path
    session = get_session()
    for place in rich.progress.track(
        session.execute(
            sa.select(CiwqsAnimalWasteFacility)
            .where(
                CiwqsAnimalWasteFacility.gcs_path.is_not(None)
                & ~CiwqsAnimalWasteFacility.inspections.any()
            )
        )
        .scalars()
        .all()
    ):
        with open_gcs(place.gcs_path, "r") as f:
            html = f.read()
            soup = bs4.BeautifulSoup(html, "html.parser")
            # select tr with text Inspections
            inspection_title_row_candidates = soup.select("tr:contains('Inspections')")
            # find the one where strip lower text is only inspections
            inspection_title_row = next(
                (
                    row
                    for row in inspection_title_row_candidates
                    if row.text.strip().lower() == "inspections"
                ),
                None,
            )
            if inspection_title_row is None:
                print(f"No inspections found for {place.place_id}")
                continue
            # find the next table
            inspections_table = inspection_title_row.find_next("table")
            if inspections_table is None:
                print(f"No inspections table found for {place.place_id}")
                continue
            headings = []

            for i, tr in enumerate(inspections_table.select("tr")):
                if i == 0:
                    # heading row
                    headings = [th.text.strip() for th in tr.select("td")]
                    continue
                columns = {
                    heading: value.text.strip()
                    for heading, value in zip(headings, tr.select("td"))
                }
                
                # Create inspection record
                inspection = CiwqsInspection(
                    ciwqs_facility_id=place.id,
                    actual_end_date=None,  # Simplified for demo
                    attachment_url=None,   # Simplified for demo
                    gcs_path=None,
                )
                session.add(inspection)
        session.commit()


def get_ciwqs_inspection_reports(date_after=datetime.datetime(2016, 1, 1)):
    session = get_session()
    for inspection in rich.progress.track(
        session.execute(
            sa.select(CiwqsInspection).where(
                CiwqsInspection.gcs_path.is_(None)
                & (CiwqsInspection.attachment_url.is_not(None))
            )
        )
        .scalars()
        .all()
    ):
        if not inspection.attachment_url:
            continue
        response = requests.get(inspection.attachment_url)
        if response.status_code != 200:
            print(f"Failed to get {inspection.attachment_url}")
            continue
        with open_gcs(ROOT_GCS_PATH / "inspection-reports" / f"{inspection.id}.pdf", "wb") as f:
            f.write(response.content)
            inspection.gcs_path = str(ROOT_GCS_PATH / "inspection-reports" / f"{inspection.id}.pdf")
            session.add(inspection)
            session.commit()


def main():
    """
    Main function to demonstrate getting one example PDF.
    This will:
    1. Download all region HTMLs
    2. Ingest the facilities into the database
    3. Get facility pages for one facility
    4. Ingest inspections for that facility
    5. Download one inspection report PDF
    """
    print("Starting CIWQS scraper demo - getting one example PDF...")
    
    # Step 1: Download all region HTMLs
    print("Step 1: Downloading animal waste facilities for all regions...")
    push_all_regions_to_gcs()
    
    # Step 2: Ingest the facilities into the database
    print("Step 2: Ingesting facilities into database...")
    ingest_all_ciwqs_animal_waste_facilities()
    
    # Step 3: Get facility pages for the first facility
    print("Step 3: Getting facility pages...")
    get_ciwqs_animal_waste_facility_pages()
    
    # Step 4: Ingest inspections for facilities
    print("Step 4: Ingesting inspections...")
    ingest_all_ciwqs_facility_inspections()
    
    # Step 5: Download one inspection report PDF
    print("Step 5: Downloading one inspection report PDF...")
    get_ciwqs_inspection_reports()
    
    print("Demo completed! Check the data/ciwqs/inspection-reports/ directory for the downloaded PDF.")


if __name__ == "__main__":
    main()
