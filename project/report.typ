#import "template.typ": *

#show: ieee.with(
  title: "The title",
  abstract: [
    This is the abstraction
  ],
  authors: (
    (
      name: "Tien Yu Lin, 5556162",
      department: [Department of Computer Science],
      organization: [University of Warwick],
    ),
  ),
  index-terms: (
    "Foo",
    "Bar"
  ),
  bibliography-file: "bibliographies.yml",
)
