"""Optional compatibility layers for migrating code written for other interfaces.

These are migration aids, frozen at their current scope: the native API (:class:`opensps.Context`
and its batch call) is where new features go.

- :mod:`opensps.compat.fsps`: a python-fsps style ``StellarPopulation`` and
  ``install_as_fsps()`` (for Prospector's ``SSPBasis`` / ``CSPSpecBasis`` / ``FastStepBasis``).
"""
