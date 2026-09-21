"""Production's catalogs, used here so training resolves names the way production does.

The fleet and the location list are package data inside `fuel_predictor`; the
lineage (unit → type → group) and the fingerprint a package must declare come
from the same place (ADR 0015). Nothing about the fleet is defined in this
repository.
"""

from dataclasses import dataclass

from fuel_predictor.application.catalog_resolution import (
    UnknownLocationError,
    UnknownVehicleError,
    resolve_location,
    resolve_vehicle,
)
from fuel_predictor.application.vehicles import VehicleLineage, catalog_fingerprint
from fuel_predictor.infrastructure.packaged_location_catalog import PackagedLocationCatalog
from fuel_predictor.infrastructure.packaged_vehicle_catalog import PackagedVehicleCatalog

# How the workbook names units that the catalog spells differently. Each entry
# is asserted against the catalog on load, so a unit renamed in production
# fails loudly here instead of becoming "tidak diketahui" in the data.
SHEET_VEHICLES: dict[str, str] = {
    "Prime Mover": "Prime Mover",
    "Truck Crane 01": "Truck Crane 01",
    "Truck Crane 02": "Truck Crane 02",
    "Whellcrane": "Wheel Crane",
    "OFT Tronton": "Oil Field Truck",
    "OFT Winch Truck": "Winch Truck",
}


@dataclass(frozen=True, slots=True)
class Catalogs:
    vehicles: PackagedVehicleCatalog
    locations: PackagedLocationCatalog

    @classmethod
    def load(cls) -> "Catalogs":
        catalogs = cls(PackagedVehicleCatalog(), PackagedLocationCatalog())
        for written, canonical in SHEET_VEHICLES.items():
            found = catalogs.vehicles.find(canonical)
            if found is None or found.name != canonical:
                raise LookupError(
                    f"SHEET_VEHICLES maps {written!r} to {canonical!r}, which production's "
                    "catalog does not list. Update the mapping or the catalog."
                )
        return catalogs

    def vehicle(self, written: str) -> str | None:
        """Canonical unit name for a workbook spelling, or None when unknown."""
        mapped = SHEET_VEHICLES.get(written.strip(), written)
        try:
            return resolve_vehicle(self.vehicles, mapped).name
        except UnknownVehicleError:
            return None

    def lineage(self, canonical: str | None) -> VehicleLineage:
        return self.vehicles.lineage_of(canonical)

    def location(self, written: str) -> str | None:
        """Catalogued spelling of a stop, or None when the name is not known."""
        try:
            return resolve_location(self.locations, written).name
        except UnknownLocationError:
            return None

    def fingerprint(self) -> str:
        return catalog_fingerprint(self.vehicles.options())
