"""
Title generation class and utilities for AQUA plots.
"""

from typing import Optional, Union
from aqua.core.util import to_list, strlist_to_phrase


class TitleBuilder:
    """
    Class to generate standardized titles for AQUA plots.

    Args:
        title (str, optional): Explicit title override.
        diagnostic (str, optional): Name of the diagnostic (e.g., 'Seasonal cycle', 'Global bias').
        variable (str, optional): Long name of the variable (e.g., 'Total precipitation rate').
        regions (str or list, optional): Region name(s) (e.g., 'global', 'North Atlantic').
        catalog (str or list, optional): Catalog name(s).
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
                 catalog: Optional[Union[str, list]] = None,
                 models: Optional[Union[str, list]] = None, 
                 exps: Optional[Union[str, list]] = None,
                 realizations: Optional[Union[str, list]] = None,
                 comparison: Optional[str] = None,
                 ref_catalog: Optional[Union[str, list]] = None,
                 ref_model: Optional[Union[str, list]] = None, 
                 ref_exp: Optional[Union[str, list]] = None,
                 timeseason: Optional[str] = None,
                 startyear: Optional[int | str] = None,
                 endyear: Optional[int | str] = None,
                 extra_info: Optional[Union[str, list]] = None,
                 ):

        self.title = title
        self.diagnostic = diagnostic
        self.variable = variable
        self.regions = regions
        self.conjunction = conjunction
        self.catalog = to_list(catalog) if catalog else []
        self.models = to_list(models) if models else []
        self.exps = to_list(exps) if exps else []
        self.realizations = to_list(realizations) if realizations else []
        self.comparison = comparison
        self.ref_catalog = to_list(ref_catalog) if ref_catalog else []
        self.ref_model = to_list(ref_model) if ref_model else []
        self.ref_exp = to_list(ref_exp) if ref_exp else []
        self.timeseason = timeseason
        self.startyear = str(startyear) if isinstance(startyear, int) else startyear
        self.endyear = str(endyear) if isinstance(endyear, int) else endyear
        self.extra_info = extra_info

    @staticmethod
    def _harmonize_lists(*lists, sep: str = " ") -> list:
        """
        Combines multiple lists element-wise, skipping empty/None values.
        """
        combined = [sep.join(filter(None, map(str, row))).strip() 
                    for row in zip(*lists)]
        return [item for item in combined if item]

    def _format_models(self) -> str | None:
        """
        Generate the models
        """
        listpart = list(filter(None, [self.catalog, self.models, self.exps]))
        listpart = self._harmonize_lists(*listpart)
        
        if listpart:
            if len(listpart) > 1:
                return "Multi-model "
            return listpart[0]
        return None

    def _format_refs(self) -> str | None:
        """
        Generate the reference
        """
        ref_listpart = list(filter(None, [self.ref_catalog, self.ref_model, self.ref_exp]))
        ref_listpart = self._harmonize_lists(*ref_listpart)

        if ref_listpart:
            ref_list_unique = list(dict.fromkeys(ref_listpart))
            return ", ".join(ref_list_unique)
        return None
        
    def _format_years(self) -> str | None:
        """
        Generate the years
        """
        if self.startyear and self.endyear:
            return f"{self.startyear}-{self.endyear}"
        if self.startyear:
            return self.startyear
        if self.endyear:
            return self.endyear
        return None

    def generate(self) -> str:
        """
        Generate the whole title
        """

        if self.title:
            return self.title

        title = ''
        if self.diagnostic:
            title += f"{self.diagnostic} "

        if self.variable:
            title += f"of {self.variable} " if self.diagnostic else f"{self.variable} "

        if self.regions:
            regions_list = to_list(self.regions)
            regions_str = strlist_to_phrase(regions_list)
            if regions_str:
                title += f"[{regions_str}] "
        
        models_part = self._format_models()
        if models_part:
            if self.variable:
                title += f"{self.conjunction} " if self.conjunction else 'for '
            title += f"{models_part} "

        if self.realizations:
            if len(self.realizations) > 1:
                title += f"Multi-realization "
            else:
                title += f"{self.realizations[0]} "

        refs_part = self._format_refs()
        if refs_part:
            if self.variable:
                title += self.comparison if self.comparison else 'relative to '
            title += refs_part

        if self.timeseason:
            title += f" {self.timeseason}"

        years = self._format_years()
        if years:
            title += f" {years}"

        if self.extra_info:
            title += f" {' '.join(to_list(self.extra_info))}"

        return title.strip()