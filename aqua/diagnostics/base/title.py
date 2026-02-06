"""
Title generation class and utilities for AQUA plots.
"""

from typing import Optional, Union
from aqua.core.util import to_list, strlist_to_phrase
from .strings import harmonize_lists


class TitleBuilder:
    """
    Class to generate standardized titles for AQUA plots.

    Args:
        title (str, optional): Explicit title override.
        diagnostic (str, optional): Name of the diagnostic (e.g., 'Seasonal cycle', 'Global bias').
        variable (str, optional): Long name of the variable (e.g., 'Total precipitation rate').
        regions (str or list, optional): Region name(s) (e.g., 'global', 'North Atlantic').
        catalogs (str or list, optional): Catalog name(s).
        models (str or list, optional): Model name(s).
        exps (str or list, optional): Experiment name(s).
        realizations (str or list, optional): Realization name(s).
        comparison (str, optional): Formulation for the comparison. Default is 'relative to'.
        ref_catalog (str or list, optional): Reference catalog name.
        ref_model (str or list, optional): Reference model name.
        ref_exp (str or list, optional): Reference experiment name.
        timeseason (str, optional): Season or month (e.g., 'JJA', 'March').
        startyear (int | str, optional): Start year.
        endyear (int | str, optional): End year.
        extra_info (str or list, optional): Extra information to be added to the title.

    Returns:
        str: The generated title.
    """

    def __init__(self, 
                 title: Optional[str] = None,
                 diagnostic: Optional[str] = None,
                 variable: Optional[str] = None,
                 regions: Optional[Union[str, list]] = None,
                 conjunction: Optional[str] = None,
                 catalogs: Optional[Union[str, list]] = None,
                 models: Optional[Union[str, list]] = None, 
                 exps: Optional[Union[str, list]] = None,
                 startyear: Optional[int | str] = None,
                 endyear: Optional[int | str] = None,
                 realizations: Optional[Union[str, list]] = None,
                 comparison: Optional[str] = None,
                 ref_catalog: Optional[Union[str, list]] = None,
                 ref_model: Optional[Union[str, list]] = None, 
                 ref_exp: Optional[Union[str, list]] = None,
                 timeseason: Optional[str] = None,
                 ref_startyear: Optional[int | str] = None,
                 ref_endyear: Optional[int | str] = None,
                 extra_info: Optional[Union[str, list]] = None,
                 ):

        self.title = title
        self.diagnostic = diagnostic
        self.variable = variable
        self.regions = regions
        self.conjunction = conjunction
        self.catalogs = to_list(catalogs) if catalogs else []
        self.models = to_list(models) if models else []
        self.exps = to_list(exps) if exps else []
        self.startyear = str(startyear) if isinstance(startyear, int) else startyear
        self.endyear = str(endyear) if isinstance(endyear, int) else endyear
        self.realizations = to_list(realizations) if realizations else []
        self.comparison = comparison
        self.ref_catalog = to_list(ref_catalog) if ref_catalog else []
        self.ref_model = to_list(ref_model) if ref_model else []
        self.ref_exp = to_list(ref_exp) if ref_exp else []
        self.timeseason = timeseason
        self.ref_startyear = str(ref_startyear) if isinstance(ref_startyear, int) else ref_startyear
        self.ref_endyear = str(ref_endyear) if isinstance(ref_endyear, int) else ref_endyear
        self.extra_info = extra_info

    def _format_models(self) -> str | None:
        """Format catalogs, models, and exps into a single models phrase.

        Returns:
            str or None: E.g. A single model string, 'Multi-model ', or None (if empty).
        """
        listpart = list(filter(None, [self.catalogs, self.models, self.exps]))
        listpart = harmonize_lists(*listpart)
        
        if listpart:
            if len(listpart) > 1:
                return "Multi-model "
            return listpart[0]
        return None

    def _format_refs(self) -> str | None:
        """Format ref_catalog, ref_model, and ref_exp into a single reference phrase.

        Returns:
            str or None: Comma-separated reference string or None (if empty).
        """
        ref_listpart = list(filter(None, [self.ref_catalog, self.ref_model, self.ref_exp]))
        ref_listpart = harmonize_lists(*ref_listpart)

        if ref_listpart:
            ref_list_unique = list(dict.fromkeys(ref_listpart))
            return ", ".join(ref_list_unique)
        return None
        
    def _format_years(self, startyear=None, endyear=None) -> str | None:
        """Format start and end year into a year range string.

        Args:
            startyear (str or int, optional): Start year.
            endyear (str or int, optional): End year.

        Returns:
            str or None: E.g. '1990-2000', a single year, or None (if both missing).
        """
        if startyear and endyear:
            return f"{startyear}-{endyear}"
        if startyear:
            return startyear
        if endyear:
            return endyear
        return None

    def generate(self) -> str:
        """Build the full title from configured components.

        Returns:
            str: The assembled title string (stripped).
        """

        if self.title:
            return self.title

        title = ''
        if self.diagnostic:
            title += f"{self.diagnostic}"

        if self.variable:
            title += f" of {self.variable}" if self.diagnostic else f" {self.variable}"

        if self.regions:
            regions_list = to_list(self.regions)
            regions_str = strlist_to_phrase(regions_list)
            if regions_str:
                title += f" [{regions_str}]"
        
        models_part = self._format_models()
        if models_part:
            if title:
                title += f" {self.conjunction}" if self.conjunction else ' for'
            title += f" {models_part}"

        if self.realizations:
            if len(self.realizations) > 1:
                title += f" Multi-realization"
            else:
                title += f" {self.realizations[0]}"

        years = self._format_years(startyear=self.startyear, endyear=self.endyear)
        if years:
            title += f" {years}"

        refs_part = self._format_refs()
        if refs_part:
            if title:
                title += f" {self.comparison}" if self.comparison else ' relative to'
            title += f" {refs_part}"

        ref_years = self._format_years(startyear=self.ref_startyear, endyear=self.ref_endyear)
        if ref_years:
            title += f" {ref_years}"

        if self.timeseason:
            title += f" {self.timeseason}"

        if self.extra_info:
            title += f" {' '.join(to_list(self.extra_info))}"

        return title.strip()