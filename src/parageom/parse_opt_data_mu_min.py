from .tasks import example
from pymor.core.pickle import load

with example.fom_minimization_data.open('rb') as fh:
    fomdata = load(fh)

rom_data_path = example.rom_minimization_data("normal", "hapod")
with rom_data_path.open('rb') as fh:
    romdata = load(fh)

print(f"FOM optimum: {fomdata['mu_min']}")
print(f"ROM optimum: {romdata['mu_N_min']}")
breakpoint()
